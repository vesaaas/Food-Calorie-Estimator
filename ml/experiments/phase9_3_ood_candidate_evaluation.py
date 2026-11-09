"""Validation-first OOD evaluation using the frozen Phase 7 v2 classifier.

This script performs bounded inference and trains only a small logistic head over
frozen classifier features. It never changes or overwrites production artifacts.
The held-out split is loaded and evaluated only when a validation candidate meets
all frozen engineering targets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/data/processed/robustness_manifest_with_openimages.csv"
PRODUCTION_TFLITE = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2.tflite"
PRODUCTION_KERAS = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_best.keras"
REPORT = ROOT / "ml/artifacts/experiments/phase9_3_ood_candidate_evaluation.json"
LABELS = (
    "pizza", "hamburger", "sushi", "steak", "ice_cream", "donuts",
    "omelette", "french_fries", "chicken_wings", "spaghetti_bolognese",
    "borek", "kebap", "sarma",
)
SUPPORTED = "supported_food"
UNSUPPORTED = "unsupported_food"
NON_FOOD = "non_food"
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TRAIN_CAPS = {SUPPORTED: 150, UNSUPPORTED: 50, NON_FOOD: 0}
TARGETS = {
    "supported_food_acceptance_min": 0.97,
    "supported_class_acceptance_min": 0.95,
    "non_food_rejection_min": 0.80,
    "unsupported_food_rejection_min": 0.50,
    "candidate_added_model_size_max_bytes": 1_000_000,
    "estimated_detector_cpu_latency_p95_max_ms": 10.0,
    "source": "project engineering targets set before candidate evaluation; not reviewer-specified",
}


def load_manifest() -> dict[str, list[dict[str, str]]]:
    splits = {name: [] for name in ("train", "validation", "test")}
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split")
            if split not in splits or row.get("duplicate_label_conflict", "false").lower() == "true":
                continue
            if row.get("triage_label") not in (SUPPORTED, UNSUPPORTED, NON_FOOD):
                continue
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            splits[split].append(row)
    for split, rows in splits.items():
        found = {row["triage_label"] for row in rows}
        if found != {SUPPORTED, UNSUPPORTED, NON_FOOD}:
            raise ValueError(f"{split} is missing a triage group: {sorted(found)}")
    _verify_split_isolation(splits)
    return splits


def _verify_split_isolation(splits: dict[str, list[dict[str, str]]]) -> None:
    owners: dict[str, dict[str, set[str]]] = {key: {} for key in ("group_id", "sha256", "dhash")}
    for split, rows in splits.items():
        for row in rows:
            for key, value in (("group_id", row.get("group_id", "")),
                               ("sha256", row.get("sha256", "")),
                               ("dhash", row.get("dhash", ""))):
                if value:
                    owners[key].setdefault(value, set()).add(split)
    leaked = {key: sum(len(value) > 1 for value in items.values())
              for key, items in owners.items()}
    leaked = {key: count for key, count in leaked.items() if count}
    if leaked:
        raise ValueError(f"manifest has exact group/hash cross-split leakage: {leaked}")


def choose_training_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        triage = row["triage_label"]
        label = row.get("food_class", "") if triage != NON_FOOD else NON_FOOD
        grouped.setdefault((triage, label), []).append(row)
    rng = random.Random(SEED)
    selected = []
    for (triage, _), members in sorted(grouped.items()):
        members.sort(key=lambda row: row["source_id"])
        cap = TRAIN_CAPS[triage]
        if cap and len(members) > cap:
            members = sorted(rng.sample(members, cap), key=lambda row: row["source_id"])
        selected.extend(members)
    return selected


def preprocess(path: str) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        size = (int(width * scale), int(height * scale))
        image = image.resize(size, Image.Resampling.BILINEAR)
        left = int(round((image.width - IMAGE_SIZE) / 2.0))
        top = int(round((image.height - IMAGE_SIZE) / 2.0))
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected preprocessed shape {array.shape} from {path}")
    return array


def batches(rows: list[dict[str, str]]):
    for start in range(0, len(rows), BATCH_SIZE):
        current = rows[start:start + BATCH_SIZE]
        yield current, np.stack([preprocess(row["path"]) for row in current])


def infer_features_and_tflite(
    rows: list[dict[str, str]], feature_model, interpreter: tf.lite.Interpreter,
    *, include_tflite: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    features, probabilities = [], []
    started = time.perf_counter()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_index, output_index = input_detail["index"], output_detail["index"]
    for _, images in batches(rows):
        feature_batch = feature_model(images, training=False).numpy()
        if not np.isfinite(feature_batch).all():
            raise ValueError("Frozen feature model returned non-finite features")
        features.append(feature_batch.astype(np.float32, copy=False))
        if include_tflite:
            # The interpreter is resized once to a fixed batch. Pad only the
            # final partial batch, then discard padded predictions.
            actual_count = len(images)
            if actual_count < BATCH_SIZE:
                tflite_images = np.zeros(
                    (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32,
                )
                tflite_images[:actual_count] = images
            else:
                tflite_images = images
            interpreter.set_tensor(input_index, tflite_images)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_index)[:actual_count]
            if not np.isfinite(scores).all():
                raise ValueError("Production TFLite classifier returned non-finite scores")
            probabilities.append(scores.astype(np.float32, copy=False))
    elapsed = time.perf_counter() - started
    feature_array = np.concatenate(features, axis=0)
    score_array = np.concatenate(probabilities, axis=0) if include_tflite else None
    return feature_array, score_array, elapsed


def groups(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    triage = np.asarray([row["triage_label"] for row in rows])
    classes = np.asarray([row.get("food_class", "") for row in rows])
    masks = {
        SUPPORTED: triage == SUPPORTED,
        UNSUPPORTED: triage == UNSUPPORTED,
        NON_FOOD: triage == NON_FOOD,
    }
    return triage, classes, masks


def metrics(rows: list[dict[str, str]], accepted: np.ndarray) -> dict:
    triage, class_names, masks = groups(rows)
    per_class = {
        label: float(accepted[masks[SUPPORTED] & (class_names == label)].mean())
        for label in LABELS
    }
    accepted_rate = float(accepted[masks[SUPPORTED]].mean())
    return {
        "supported_food_acceptance": accepted_rate,
        "supported_class_acceptance": per_class,
        "supported_class_floor": min(per_class.values()),
        "unsupported_food_rejection": float((~accepted[masks[UNSUPPORTED]]).mean()),
        "non_food_rejection": float((~accepted[masks[NON_FOOD]]).mean()),
        "inconclusive_rate": 0.0,
        "samples": {
            "supported_food": int(masks[SUPPORTED].sum()),
            "unsupported_food": int(masks[UNSUPPORTED].sum()),
            "non_food": int(masks[NON_FOOD].sum()),
        },
        "confusion_by_group": {
            name: {"accepted": int(accepted[mask].sum()),
                   "rejected": int((~accepted[mask]).sum())}
            for name, mask in masks.items()
        },
        "supported_food_acceptance_by_class": per_class,
    }


def meets_targets(result: dict) -> bool:
    return (
        result["supported_food_acceptance"] >= TARGETS["supported_food_acceptance_min"]
        and result["supported_class_floor"] >= TARGETS["supported_class_acceptance_min"]
        and result["non_food_rejection"] >= TARGETS["non_food_rejection_min"]
        and result["unsupported_food_rejection"] >= TARGETS["unsupported_food_rejection_min"]
    )


def select_threshold(rows: list[dict[str, str]], ood_score: np.ndarray) -> dict:
    # An image is accepted when its score is at least the threshold.
    candidates = np.unique(np.concatenate(([0.0], ood_score)))
    best = None
    best_score = None
    for threshold in candidates:
        result = metrics(rows, ood_score >= threshold)
        result["threshold"] = float(threshold)
        result["meets_targets"] = meets_targets(result)
        normalized_attainment = [
            result["supported_food_acceptance"] / TARGETS["supported_food_acceptance_min"],
            result["supported_class_floor"] / TARGETS["supported_class_acceptance_min"],
            result["non_food_rejection"] / TARGETS["non_food_rejection_min"],
            result["unsupported_food_rejection"] / TARGETS["unsupported_food_rejection_min"],
        ]
        # When no point meets every target, use mean normalized target
        # attainment to expose the best tradeoff instead of a degenerate
        # all-rejected point that maximizes only OOD rejection.
        attainment = float(np.mean(normalized_attainment))
        result["validation_target_attainment_score"] = attainment
        order = (
            result["meets_targets"], attainment, min(normalized_attainment),
            result["non_food_rejection"], result["unsupported_food_rejection"],
            result["supported_food_acceptance"], float(threshold),
        )
        if best is None or order > best_score:
            best, best_score = result, order
    return best


def failure_examples(rows: list[dict[str, str]], accepted: np.ndarray,
                     score: np.ndarray, *, group: str, accepted_errors: bool) -> list[dict]:
    selected = [i for i, row in enumerate(rows)
                if row["triage_label"] == group and bool(accepted[i]) == accepted_errors]
    selected.sort(key=lambda i: float(score[i]), reverse=accepted_errors)
    return [{
        "source_id": rows[i].get("source_id"),
        "food_class": rows[i].get("food_class"),
        "scene_tags": rows[i].get("scene_tags"),
        "score": float(score[i]),
        "path": rows[i]["path"],
    } for i in selected[:10]]


def target_confusion(score_matrix: np.ndarray, rows: list[dict[str, str]]) -> dict:
    labels = np.asarray([row.get("food_class", "") for row in rows])
    top = np.argmax(score_matrix, axis=1)
    return {
        "prediction_labels": list(LABELS),
        "non_food_top_predictions": {
            LABELS[index]: int(np.sum((np.asarray([r["triage_label"] for r in rows]) == NON_FOOD) & (top == index)))
            for index in range(len(LABELS))
        },
        "unsupported_food_top_predictions": {
            LABELS[index]: int(np.sum((np.asarray([r["triage_label"] for r in rows]) == UNSUPPORTED) & (top == index)))
            for index in range(len(LABELS))
        },
        "supported_top1_accuracy_by_true_class": {
            name: float(np.mean(top[(labels == name)] == LABELS.index(name)))
            for name in LABELS if np.any(labels == name)
        },
    }


def main() -> None:
    for path in (MANIFEST, PRODUCTION_TFLITE, PRODUCTION_KERAS):
        if not path.is_file():
            raise FileNotFoundError(path)

    splits = load_manifest()
    train_rows = choose_training_sample(splits["train"])
    validation_rows = splits["validation"]
    counts = {
        split: {group: sum(row["triage_label"] == group for row in rows)
                for group in (SUPPORTED, UNSUPPORTED, NON_FOOD)}
        for split, rows in splits.items()
    }

    model = tf.keras.models.load_model(PRODUCTION_KERAS, compile=False)
    if tuple(model.output_shape) != (None, len(LABELS)):
        raise ValueError(f"Unexpected production Keras output shape {model.output_shape}")
    feature_model = tf.keras.Model(
        model.input,
        model.get_layer("class_probabilities").input,
        name="frozen_phase7_v2_features",
    )
    feature_size = int(feature_model.output_shape[-1])

    interpreter = tf.lite.Interpreter(model_path=str(PRODUCTION_TFLITE), num_threads=2)
    input_info = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_info["index"], [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], strict=False)
    interpreter.allocate_tensors()

    train_features, _, train_seconds = infer_features_and_tflite(
        train_rows, feature_model, interpreter, include_tflite=False,
    )
    validation_features, validation_probabilities, validation_seconds = infer_features_and_tflite(
        validation_rows, feature_model, interpreter, include_tflite=True,
    )
    assert validation_probabilities is not None

    # Softmax-only OOD baseline. Threshold is selected on validation subject to
    # the predeclared targets; no test scores are loaded at this stage.
    maximum_probability = validation_probabilities.max(axis=1)
    baseline = select_threshold(validation_rows, maximum_probability)
    baseline_fixed_055 = metrics(validation_rows, maximum_probability >= 0.55)
    baseline_accepted = maximum_probability >= baseline["threshold"]

    train_labels = np.asarray([int(row["triage_label"] == SUPPORTED) for row in train_rows], dtype=np.int32)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    detector = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=500,
        random_state=SEED,
    )
    fit_started = time.perf_counter()
    detector.fit(train_scaled, train_labels)
    fit_seconds = time.perf_counter() - fit_started
    supported_column = list(detector.classes_).index(1)
    supported_probability = detector.predict_proba(validation_scaled)[:, supported_column]
    frozen_detector = select_threshold(validation_rows, supported_probability)
    detector_accepted = supported_probability >= frozen_detector["threshold"]

    # Estimate detector-only inference cost with cached validation embeddings.
    probe = validation_features[:min(512, len(validation_features))]
    exported_scale = np.asarray(scaler.scale_, dtype=np.float32)
    exported_mean = np.asarray(scaler.mean_, dtype=np.float32)
    exported_weight = np.asarray(detector.coef_[0], dtype=np.float32)
    exported_bias = float(detector.intercept_[0])
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        ((probe - exported_mean) / exported_scale) @ exported_weight + exported_bias
        timings.append((time.perf_counter() - start) * 1000 / len(probe))
    latency = {
        "p50_ms_per_embedding": float(statistics.median(timings)),
        "p95_ms_per_embedding": float(np.percentile(timings, 95)),
        "method": "NumPy standardized linear score, host CPU, 20 batches of up to 512 cached embeddings; excludes feature extraction",
    }
    model_bytes = int(exported_scale.nbytes + exported_mean.nbytes + exported_weight.nbytes + 4)

    report = {
        "phase": "9.3",
        "evaluation_status": "validation evaluated; test pending validation acceptance",
        "targets": TARGETS,
        "data": {
            "manifest": str(MANIFEST),
            "manifest_split_integrity": (
                "locally checked group_id, sha256, and exact dhash collisions across splits; "
                "manifest QA reports source-group, exact-hash, and dHash-distance<=5 "
                "isolation passed"
            ),
            "conflict_rows_excluded": True,
            "split_counts": counts,
            "training_sample_counts": {
                group: sum(row["triage_label"] == group for row in train_rows)
                for group in (SUPPORTED, UNSUPPORTED, NON_FOOD)
            },
            "unsupported_food_training_cap_per_class": TRAIN_CAPS[UNSUPPORTED],
            "supported_training_cap_per_class": TRAIN_CAPS[SUPPORTED],
            "non_food_training_cap": "all available train examples",
            "frozen_supported_test_membership": "existing 2,719 supported-food examples preserved in manifest; not used for threshold or candidate selection",
        },
        "production_model": {
            "tflite": str(PRODUCTION_TFLITE),
            "keras_checkpoint": str(PRODUCTION_KERAS),
            "feature_tensor": "input to class_probabilities (frozen pooled MobileNetV3 features)",
            "feature_dimensions": feature_size,
            "tflite_score_preprocessing": "EXIF transpose, RGB, shortest edge 256, bilinear, center crop 224, float32 [0,255]",
        },
        "softmax_score_baseline": {
            "score": "maximum of the 13 production TFLite softmax scores",
            "validation_selected_operating_point": baseline,
            "existing_0_55_threshold_validation_metrics": baseline_fixed_055,
            "high_confidence_non_food_false_accepts_at_selected_threshold": failure_examples(
                validation_rows, baseline_accepted, maximum_probability,
                group=NON_FOOD, accepted_errors=True,
            ),
            "unsupported_food_false_accepts_at_selected_threshold": failure_examples(
                validation_rows, baseline_accepted, maximum_probability,
                group=UNSUPPORTED, accepted_errors=True,
            ),
            "top1_failure_breakdown": target_confusion(validation_probabilities, validation_rows),
        },
        "frozen_feature_logistic_detector": {
            "training_method": "StandardScaler + class-balanced LogisticRegression(C=1.0, liblinear) over frozen 576-d features; no classifier/backbone retraining",
            "threshold_selection": (
                "selected on validation only: among points meeting all targets, "
                "prioritize normalized target attainment; if none meet all targets, "
                "choose the point with greatest mean normalized target attainment"
            ),
            "validation_operating_point": frozen_detector,
            "model_size_bytes_estimate": model_bytes,
            "model_size_note": "float32 scale, mean, linear weights, and one bias; assumes embeddings are exposed from the existing classifier graph",
            "detector_only_host_latency": latency,
            "fit_seconds": float(fit_seconds),
            "feature_extraction_seconds": {
                "training_sample": float(train_seconds),
                "validation": float(validation_seconds),
            },
            "failure_examples": {
                "non_food_false_accepts": failure_examples(
                    validation_rows, detector_accepted, supported_probability,
                    group=NON_FOOD, accepted_errors=True,
                ),
                "unsupported_food_false_accepts": failure_examples(
                    validation_rows, detector_accepted, supported_probability,
                    group=UNSUPPORTED, accepted_errors=True,
                ),
                "supported_food_false_rejections": failure_examples(
                    validation_rows, detector_accepted, supported_probability,
                    group=SUPPORTED, accepted_errors=False,
                ),
            },
        },
    }

    validation_pass = frozen_detector["meets_targets"]
    report["frozen_feature_logistic_detector"]["validation_meets_all_targets"] = validation_pass
    if validation_pass:
        # Threshold and candidate are frozen before this single test pass.
        test_rows = splits["test"]
        test_features, _, test_seconds = infer_features_and_tflite(
            test_rows, feature_model, interpreter, include_tflite=False,
        )
        test_scaled = scaler.transform(test_features)
        test_probability = detector.predict_proba(test_scaled)[:, supported_column]
        test_accepted = test_probability >= frozen_detector["threshold"]
        report["evaluation_status"] = "selected candidate threshold frozen on validation; held-out test evaluated once"
        report["frozen_feature_logistic_detector"]["held_out_test_operating_point"] = metrics(test_rows, test_accepted)
        report["frozen_feature_logistic_detector"]["held_out_test_operating_point"]["threshold_frozen_from_validation"] = frozen_detector["threshold"]
        report["frozen_feature_logistic_detector"]["held_out_test_operating_point"]["test_used_for_candidate_selection"] = False
        report["frozen_feature_logistic_detector"]["held_out_test_feature_extraction_seconds"] = float(test_seconds)
    else:
        report["frozen_feature_logistic_detector"]["held_out_test_operating_point"] = None
        report["frozen_feature_logistic_detector"]["held_out_test_reason_not_run"] = (
            "Validation targets were not met; held-out test was not evaluated."
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "report": str(REPORT),
        "validation_rows": len(validation_rows),
        "sampled_train_rows": len(train_rows),
        "softmax_validation": baseline,
        "frozen_feature_validation": frozen_detector,
        "validation_targets_met": validation_pass,
        "test_evaluated": validation_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
