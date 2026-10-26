"""Train isolated three-way triage or 13-class robustness candidates from a CSV manifest.

This command does not alter the mobile deployment artifact. All output paths must
be new. Training and checkpoint selection use train/validation only; test is
evaluated once after restoring the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
    from .robustness_data import TRIAGE_CLASSES
else:
    from dataset import PHASE5_CLASSES
    from robustness_data import TRIAGE_CLASSES


IMAGE_SIZE = 224
DEFAULT_INIT = Path(__file__).resolve().parents[1] / "artifacts/mobilenet_v3_small_phase7_v2_best.keras"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts/experiments"


def load_rows(manifest: Path, task: str) -> tuple[dict[str, list[dict]], tuple[str, ...]]:
    classes = TRIAGE_CLASSES if task == "triage" else PHASE5_CLASSES
    rows_by_split = {split: [] for split in ("train", "validation", "test")}
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in rows_by_split:
                continue
            if row.get("duplicate_label_conflict", "false").lower() == "true":
                continue
            label = row.get("triage_label") if task == "triage" else row.get("food_class")
            path = Path(row["path"]).expanduser()
            if not label or not path.is_file():
                continue
            if label in classes:
                row["label"] = label
                row["path"] = str(path.resolve())
                rows_by_split[row["split"]].append(row)
    for split, rows in rows_by_split.items():
        found = {row["label"] for row in rows}
        missing = set(classes) - found
        if missing:
            raise ValueError(f"{split} split is missing required labels: {sorted(missing)}")
        if task == "triage":
            food_classes = {row.get("food_class") for row in rows if row["label"] == "supported_food"}
            missing_foods = set(PHASE5_CLASSES) - food_classes
            if missing_foods:
                raise ValueError(f"{split} split lacks supported-food metadata: {sorted(missing_foods)}")
    return rows_by_split, classes


def _image_array(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = IMAGE_SIZE / min(image.size)
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BILINEAR)
        left, top = (image.width - IMAGE_SIZE) // 2, (image.height - IMAGE_SIZE) // 2
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        return np.asarray(image, dtype=np.float32)


def _dataset(rows: list[dict], class_index: dict[str, int], *, batch_size: int,
             training: bool, seed: int) -> tf.data.Dataset:
    def generate():
        ordered = list(rows)
        if training:
            random.Random(seed).shuffle(ordered)
        counts = {name: sum(row["label"] == name for row in rows) for name in class_index}
        weights = {name: len(rows) / (len(class_index) * count) for name, count in counts.items()}
        mean_weight = sum(weights.values()) / len(weights)
        weights = {name: value / mean_weight for name, value in weights.items()}
        for row in ordered:
            yield (
                _image_array(row["path"]),
                np.int32(class_index[row["label"]]),
                np.float32(weights[row["label"]]),
            )
    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=(
            tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32),
            tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.float32),
        ),
    )
    if training:
        dataset = dataset.shuffle(min(len(rows), 10000), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(1)


def build_model(task: str, initial_checkpoint: Path | None):
    if task == "classifier" and initial_checkpoint:
        base_model = tf.keras.models.load_model(initial_checkpoint, compile=False)
        if tuple(base_model.output_shape) != (None, len(PHASE5_CLASSES)):
            raise ValueError("Initial classifier checkpoint must have 13 outputs")
        inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image_rgb_0_255")
        x = _augmentation_stack(inputs)
        model = tf.keras.Model(inputs, base_model(x), name="mobilenet_v3_small_classifier_candidate")
    else:
        inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image_rgb_0_255")
        x = _augmentation_stack(inputs)
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
            weights="imagenet", include_preprocessing=True, pooling="avg",
        )
        x = backbone(x)
        x = tf.keras.layers.Dropout(0.25, name="robust_head_dropout")(x)
        classes = len(TRIAGE_CLASSES) if task == "triage" else len(PHASE5_CLASSES)
        outputs = tf.keras.layers.Dense(classes, activation="softmax", name="robust_probabilities")(x)
        model = tf.keras.Model(inputs, outputs, name=f"mobilenet_v3_small_{task}_candidate")
    return model


def _augmentation_stack(inputs):
    x = tf.keras.layers.RandomFlip("horizontal", seed=41, name="aug_flip")(inputs)
    x = tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=42, name="aug_rotation")(x)
    x = tf.keras.layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect", seed=43, name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=44, name="aug_contrast")(x)
    return tf.keras.layers.RandomBrightness(
        0.12, value_range=(0, 255), seed=45, name="aug_brightness",
    )(x)


def _predict(model, rows: list[dict], classes: tuple[str, ...], *, batch_size: int):
    class_index = {name: index for index, name in enumerate(classes)}
    dataset = _dataset(rows, class_index, batch_size=batch_size, training=False, seed=42)
    probabilities = model.predict(dataset, verbose=0)
    labels = np.asarray([class_index[row["label"]] for row in rows], dtype=np.int64)
    return labels, probabilities


def evaluate(model, rows: list[dict], classes: tuple[str, ...], *, batch_size: int) -> dict:
    labels, probabilities = _predict(model, rows, classes, batch_size=batch_size)
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    per_class = {}
    for index, label in enumerate(classes):
        chosen = labels == index
        correct = predictions[chosen] == index
        per_class[label] = {
            "samples": int(chosen.sum()), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "confidence_correct": _summary(probabilities[chosen][correct, index]),
            "confidence_incorrect": _summary(probabilities[chosen][~correct].max(axis=1)),
            "low_confidence_correct_below_0_55": int(np.sum(correct & (probabilities[chosen].max(axis=1) < 0.55))),
            "high_confidence_incorrect_at_least_0_55": int(np.sum(~correct & (probabilities[chosen].max(axis=1) >= 0.55))),
        }
    prediction_rows = []
    for index, row in enumerate(rows):
        predicted = int(predictions[index])
        prediction_rows.append({
            "index": index, "path": row["path"], "true_class": classes[int(labels[index])],
            "predicted_class": classes[predicted],
            "maximum_softmax_confidence": float(probabilities[index, predicted]),
            "correct": bool(predicted == int(labels[index])),
        })
    return {
        "samples": len(rows), "accuracy": float(np.mean(predictions == labels)),
        "ece": _ece(labels, predictions, probabilities),
        "per_class": per_class,
        "predictions": prediction_rows,
        "confusion_matrix": {"labels": list(classes), "rows_true_columns_predicted": matrix.tolist()},
    }


def _ece(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray,
         bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = labels == predictions
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        chosen = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if chosen.any():
            total += chosen.mean() * abs(float(correct[chosen].mean()) - float(confidence[chosen].mean()))
    return float(total)


def select_triage_threshold(model, validation_rows: list[dict], *, batch_size: int) -> dict:
    labels, probabilities = _predict(model, validation_rows, TRIAGE_CLASSES, batch_size=batch_size)
    supported, unsupported, non_food = (TRIAGE_CLASSES.index(name) for name in TRIAGE_CLASSES)
    best = None
    best_any = None
    candidates = np.linspace(0.20, 0.95, 76)
    for threshold in candidates:
        predicted = probabilities.argmax(axis=1)
        confident = probabilities.max(axis=1) >= threshold
        supported_mask = labels == supported
        per_supported = [
            float(np.mean(predicted[supported_mask & (labels == class_index)] == supported
                          & confident[supported_mask & (labels == class_index)]))
            for class_index in (supported,)
        ]
        # Metadata food_class supports the required 13-class preservation check.
        class_pass = {}
        for name in PHASE5_CLASSES:
            mask = np.asarray([row.get("food_class") == name for row in validation_rows])
            class_pass[name] = float(np.mean((predicted[mask] == supported) & confident[mask])) if mask.any() else None
        supported_pass = float(np.mean((predicted[supported_mask] == supported) & confident[supported_mask]))
        nonfood_mask = labels == non_food
        unsupported_mask = labels == unsupported
        nonfood_rejection = float(np.mean((predicted[nonfood_mask] == non_food) & confident[nonfood_mask])) if nonfood_mask.any() else None
        unsupported_rejection = float(np.mean((predicted[unsupported_mask] == unsupported) & confident[unsupported_mask])) if unsupported_mask.any() else None
        class_floor = min(value for value in class_pass.values() if value is not None) if any(v is not None for v in class_pass.values()) else None
        feasible = (
            supported_pass >= 0.98 and class_floor is not None and class_floor >= 0.95
            and nonfood_rejection is not None and nonfood_rejection >= 0.95
            and unsupported_rejection is not None and unsupported_rejection >= 0.90
        )
        row = {
            "threshold": float(threshold), "supported_pass_rate": supported_pass,
            "supported_class_pass_rates": class_pass, "nonfood_rejection_rate": nonfood_rejection,
            "unsupported_food_rejection_rate": unsupported_rejection,
            "inconclusive_rate": float(np.mean(~confident)), "meets_targets": feasible,
        }
        score = min(
            supported_pass / 0.98,
            (class_floor or 0.0) / 0.95,
            (nonfood_rejection or 0.0) / 0.95,
            (unsupported_rejection or 0.0) / 0.90,
        )
        row["validation_target_attainment_score"] = float(score)
        if best_any is None or (score, -row["inconclusive_rate"]) > (
            best_any["validation_target_attainment_score"], -best_any["inconclusive_rate"]
        ):
            best_any = row
        # Prefer a feasible threshold with least inconclusive coverage loss.
        if feasible and (best is None or row["inconclusive_rate"] < best["inconclusive_rate"]):
            best = row
    if best is None:
        return {"selected": best_any, "meets_all_targets": False,
                "reason": "No validation threshold meets all declared targets; selected best target-attainment point",
                "selection_data": "validation_only"}
    return {"selected": best, "meets_all_targets": True, "selection_data": "validation_only"}


def evaluate_triage(model, rows: list[dict], *, batch_size: int, threshold: float) -> dict:
    labels, probabilities = _predict(model, rows, TRIAGE_CLASSES, batch_size=batch_size)
    predictions = probabilities.argmax(axis=1)
    confident = probabilities.max(axis=1) >= threshold
    supported = TRIAGE_CLASSES.index("supported_food")
    unsupported = TRIAGE_CLASSES.index("unsupported_food")
    non_food = TRIAGE_CLASSES.index("non_food")
    supported_mask, unsupported_mask, nonfood_mask = labels == supported, labels == unsupported, labels == non_food
    class_rates = {}
    for label in PHASE5_CLASSES:
        mask = np.asarray([row.get("food_class") == label for row in rows])
        class_rates[label] = float(np.mean((predictions[mask] == supported) & confident[mask])) if mask.any() else None
    subgroup_results = {}
    for subgroup_key, getter in (
        ("source_dataset", lambda row: row.get("source_dataset", "unknown")),
        ("scene_tag", lambda row: row.get("scene_tags", "").split(";")[0] or "unlabeled"),
    ):
        results = {}
        for index, row in enumerate(rows):
            key = getter(row)
            record = results.setdefault(key, {"samples": 0, "correct_triage": 0, "rejected_or_inconclusive": 0})
            record["samples"] += 1
            record["correct_triage"] += int(predictions[index] == labels[index] and confident[index])
            record["rejected_or_inconclusive"] += int(not confident[index])
        for record in results.values():
            record["correct_triage_rate"] = record["correct_triage"] / record["samples"]
            record["inconclusive_rate"] = record["rejected_or_inconclusive"] / record["samples"]
        subgroup_results[subgroup_key] = results
    confidence_by_class = {}
    max_confidence = probabilities.max(axis=1)
    for index, name in enumerate(TRIAGE_CLASSES):
        mask = labels == index
        confidence_by_class[name] = _summary(max_confidence[mask]) if mask.any() else _summary(np.array([]))
    return {
        "threshold_frozen_from_validation": threshold,
        "supported_pass_rate": float(np.mean((predictions[supported_mask] == supported) & confident[supported_mask])) if supported_mask.any() else None,
        "supported_class_pass_rates": class_rates,
        "unsupported_food_rejection_rate": float(np.mean((predictions[unsupported_mask] == unsupported) & confident[unsupported_mask])) if unsupported_mask.any() else None,
        "non_food_rejection_rate": float(np.mean((predictions[nonfood_mask] == non_food) & confident[nonfood_mask])) if nonfood_mask.any() else None,
        "inconclusive_rate": float(np.mean(~confident)),
        "confidence_distribution_by_true_class": confidence_by_class,
        "subgroup_results": subgroup_results,
        "confusion_matrix": evaluate(model, rows, TRIAGE_CLASSES, batch_size=batch_size)["confusion_matrix"],
        "test_split_used_for_threshold_selection": False,
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def run(*, task: str, manifest: Path, checkpoint: Path, report: Path,
        initial_checkpoint: Path | None = None, epochs: int = 8,
        batch_size: int = 24, learning_rate: float = 2e-5, seed: int = 42) -> dict:
    tflite_path = checkpoint.with_suffix(".tflite")
    for output in (checkpoint, report, tflite_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite experimental output: {output}")
    if task not in ("triage", "classifier") or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid task or training parameters")
    rows, classes = load_rows(manifest, task)
    if task == "triage" and initial_checkpoint:
        raise ValueError("Triage training uses ImageNet initialization, not the 13-class checkpoint")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model = build_model(task, initial_checkpoint)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()
    history = model.fit(
        _dataset(rows["train"], {name: i for i, name in enumerate(classes)},
                 batch_size=batch_size, training=True, seed=seed),
        validation_data=_dataset(rows["validation"], {name: i for i, name in enumerate(classes)},
                                 batch_size=batch_size, training=False, seed=seed),
        epochs=epochs, callbacks=[tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_accuracy", mode="max", save_best_only=True,
            save_weights_only=False,
        )], verbose=2,
    )
    best = tf.keras.models.load_model(checkpoint, compile=False)
    training_duration_seconds = time.perf_counter() - training_started
    triage_threshold = select_triage_threshold(best, rows["validation"], batch_size=batch_size) if task == "triage" else None
    selected_gate = triage_threshold.get("selected") if triage_threshold else None
    result = {
        "task": task, "classes": list(classes), "manifest": str(manifest.resolve()),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "initial_checkpoint": str(initial_checkpoint.resolve()) if initial_checkpoint else None,
        "best_epoch": int(np.argmax(history.history["val_accuracy"])) + 1,
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "training_duration_seconds": float(training_duration_seconds),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "held_out_test": evaluate(best, rows["test"], classes, batch_size=batch_size),
        "triage_validation_threshold": triage_threshold,
        "triage_test_operating_metrics": (
            evaluate_triage(best, rows["test"], batch_size=batch_size, threshold=selected_gate["threshold"])
            if selected_gate else None
        ),
        "checkpoint": str(checkpoint.resolve()), "tensorflow_version": tf.__version__,
        "augmentations": ["horizontal flip", "small rotation", "small zoom", "contrast", "brightness"],
        "keras_model_size_bytes": checkpoint.stat().st_size,
    }
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(best)
        tflite_bytes = converter.convert()
        tflite_path.write_bytes(tflite_bytes)
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        input_data = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
        for _ in range(5):
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()
        latencies = []
        for _ in range(30):
            started = time.perf_counter()
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()
            latencies.append((time.perf_counter() - started) * 1000)
        result["tflite"] = {
            "path": str(tflite_path.resolve()), "size_bytes": len(tflite_bytes),
            "input_shape": input_detail["shape"].tolist(),
            "output_shape": output_detail["shape"].tolist(),
            "cpu_latency_ms_batch_1_median": float(np.median(latencies)),
            "cpu_latency_ms_batch_1_p90": float(np.percentile(latencies, 90)),
            "latency_method": "TensorFlow Lite CPU, one thread, zero input, 5 warmups + 30 runs",
        }
    except Exception as error:
        result["tflite"] = {"export_error": f"{type(error).__name__}: {error}"}
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                     prefix=f".{report.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temp.replace(report)
    finally:
        temp.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("triage", "classifier"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = run(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Robustness candidate training failed: {error}\n")
    print(json.dumps({key: result[key] for key in (
        "task", "counts", "best_epoch", "best_validation_accuracy", "held_out_test", "checkpoint",
    )}, indent=2))


if __name__ == "__main__":
    main()
