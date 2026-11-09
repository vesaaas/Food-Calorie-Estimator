"""One-shot held-out evaluation for the frozen Phase 9.4 14-class candidate.

Preflight reads manifest metadata only. Test image contents are accessed only
after all checkpoint, mapping, report, and membership integrity checks pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "ml/experiments"
sys.path.insert(0, str(EXPERIMENTS))
import train_phase9_3_14class as runner  # noqa: E402

DEFAULT_CONFIG = EXPERIMENTS / "phase9_3_14class_config.json"
DEFAULT_OUTPUT = ROOT / "ml/artifacts/experiments/phase9_4_14class_heldout_evaluation"
SUPPORTED = "supported_food"
NON_FOOD = "non_food"
UNSUPPORTED = "unsupported_food"
FROZEN_POLICY = "argmax over the 14 outputs; accept indices 0-12 as supported food; reject index 13 as non_food; no confidence threshold"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _membership(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row.get("source_dataset", ""), row.get("source_id", ""),
            row.get("food_class", ""), row.get("sha256", ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config_path: Path, output_dir: Path) -> dict:
    """Verify frozen artifacts and manifest membership without touching images."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classes = tuple(config["classes"])
    mapping = config["class_to_idx"]
    expected_mapping = {label: index for index, label in enumerate(classes)}
    if len(classes) != 14 or mapping != expected_mapping or classes[-1] != "non_food":
        raise ValueError("Configured 14-class mapping is not the frozen 0-12 food, 13 non_food order")

    candidate_path = ROOT / "ml/artifacts/experiments/phase9_3_14class_finetune/phase9_3_14class_finetune_best.keras"
    fine_report_path = candidate_path.parent / "fine_tune_report.json"
    head_report_path = ROOT / "ml/artifacts/experiments/phase9_3_14class/training_report.json"
    calibration_path = ROOT / "ml/artifacts/experiments/phase9_3_14class_policy_calibration/policy_calibration_report.json"
    production_path = ROOT / config["initial_checkpoint"]
    mapping_path = candidate_path.parent / "class_mapping.json"
    contract_path = ROOT / config["production_tflite_contract"]
    tflite_verification_path = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_tflite_verification.json"
    manifest_path = ROOT / config["manifest"]
    reference_manifest_path = ROOT / config["supported_manifest_for_test_membership_check"]
    required = (candidate_path, fine_report_path, head_report_path, calibration_path,
                production_path, mapping_path, contract_path, tflite_verification_path,
                manifest_path, reference_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required pre-test artifact(s) missing: {missing}")

    output_dir = output_dir.expanduser().resolve()
    artifact_root = (ROOT / "ml/artifacts/experiments").resolve()
    try:
        output_dir.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(f"Output must be under {artifact_root}") from error
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {output_dir}")

    fine_report = json.loads(fine_report_path.read_text(encoding="utf-8"))
    head_report = json.loads(head_report_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    saved_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tflite_reference = json.loads(tflite_verification_path.read_text(encoding="utf-8"))

    if Path(fine_report.get("selected_checkpoint", "")).resolve() != candidate_path.resolve():
        raise ValueError("Fine-tuning report does not select the requested frozen checkpoint")
    if fine_report.get("held_out_test", {}).get("evaluated") is not False:
        raise ValueError("Fine-tuning report does not confirm test was previously skipped")
    if fine_report.get("status") != "validation_targets_not_met":
        raise ValueError("Fine-tuning/validation report status is unexpected")
    if fine_report.get("classes") != list(classes) or fine_report.get("class_to_idx") != mapping:
        raise ValueError("Fine-tuning report class mapping differs from frozen policy")
    if head_report.get("held_out_test", {}).get("evaluated") is not False:
        raise ValueError("Head-only report does not confirm test was previously skipped")
    if calibration.get("validation_integrity", {}).get("held_out_test_images_read_or_evaluated") is not False:
        raise ValueError("Validation policy report does not confirm held-out test was untouched")
    if calibration.get("status") != "no_policy_met_all_targets_on_selection":
        raise ValueError("Expected recorded validation decision-policy result is missing")
    if saved_mapping.get("classes") != list(classes) or saved_mapping.get("class_to_idx") != mapping:
        raise ValueError("Saved candidate class mapping differs from frozen policy")

    supported_labels = list(classes[:-1])
    if contract.get("labels") != supported_labels:
        raise ValueError("Production preprocessing contract labels do not match candidate food indices")
    input_contract = contract.get("input", {})
    if (input_contract.get("tensor_shape") != [1, 224, 224, 3]
            or input_contract.get("color") != "RGB"
            or input_contract.get("dtype") != "float32"
            or input_contract.get("pixel_range") != [0, 255]
            or "shortest edge to 256" not in input_contract.get("resize", "")
            or "center crop 224x224" not in input_contract.get("resize", "")
            or "includes Keras MobileNetV3 rescaling" not in input_contract.get("normalization", "")):
        raise ValueError("Production preprocessing contract differs from the required RGB [0,255] contract")
    if tflite_reference.get("classes") != supported_labels or tflite_reference.get("tflite_test_accuracy") != 0.8812063258550937:
        raise ValueError("Deployed TFLite reference artifact is inconsistent; keep it separate from Keras metrics")

    all_rows = _read_csv(manifest_path)
    test_rows = [row for row in all_rows
                 if row.get("split") == "test"
                 and row.get("duplicate_label_conflict", "false").lower() != "true"]
    if any(row.get("triage_label") not in (SUPPORTED, NON_FOOD, UNSUPPORTED) for row in test_rows):
        raise ValueError("Test manifest contains an unknown evaluation group")
    counts = {
        "supported_food": sum(row["triage_label"] == SUPPORTED for row in test_rows),
        "non_food": sum(row["triage_label"] == NON_FOOD for row in test_rows),
        "unsupported_food_evaluation_only": sum(row["triage_label"] == UNSUPPORTED for row in test_rows),
    }
    expected_counts = config["expected_manifest_counts_after_conflict_exclusion"]
    target_counts = {
        "supported_food": expected_counts["test_supported_food_frozen"],
        "non_food": expected_counts["test_non_food"],
        "unsupported_food_evaluation_only": expected_counts["test_unsupported_food_evaluation_only"],
    }
    if counts != target_counts or counts != {
            "supported_food": 2719, "non_food": 529,
            "unsupported_food_evaluation_only": 22750}:
        raise ValueError(f"Held-out manifest counts mismatch: expected={target_counts}, actual={counts}")

    reference_rows = [row for row in _read_csv(reference_manifest_path)
                      if row.get("split") == "test"
                      and row.get("triage_label") == SUPPORTED
                      and row.get("duplicate_label_conflict", "false").lower() != "true"]
    candidate_supported_rows = [row for row in test_rows if row["triage_label"] == SUPPORTED]
    from collections import Counter
    if Counter(map(_membership, reference_rows)) != Counter(map(_membership, candidate_supported_rows)):
        raise ValueError("Frozen supported-food held-out membership differs from the existing 13-class test set")
    if len({row.get("food_class") for row in candidate_supported_rows}) != 13:
        raise ValueError("Held-out supported-food cohort does not contain all 13 classes")
    if any(not row.get("food_class") for row in test_rows if row["triage_label"] == UNSUPPORTED):
        raise ValueError("Unsupported-food test rows lost their distinct food labels")
    if any(row.get("source_dataset") != "Open Images V7"
           for row in test_rows if row["triage_label"] == NON_FOOD):
        raise ValueError("Non-food test rows are not consistently from Open Images V7")

    # Check tensor contracts and final activation without decoding any image.
    candidate_model = tf.keras.models.load_model(candidate_path, compile=False)
    production_model = tf.keras.models.load_model(production_path, compile=False)
    if tuple(candidate_model.input_shape) != (None, 224, 224, 3) or tuple(candidate_model.output_shape) != (None, 14):
        raise ValueError(f"Candidate model I/O shape mismatch: {candidate_model.input_shape}, {candidate_model.output_shape}")
    if tuple(production_model.input_shape) != (None, 224, 224, 3) or tuple(production_model.output_shape) != (None, 13):
        raise ValueError(f"Production Keras baseline I/O shape mismatch: {production_model.input_shape}, {production_model.output_shape}")
    candidate_head = candidate_model.get_layer("phase9_3_class_probabilities")
    if candidate_head.activation.__name__ != "softmax":
        raise ValueError("Candidate model output is not the expected softmax head")
    feature_extractor = candidate_model.get_layer("phase9_3_frozen_feature_extractor")
    backbone = feature_extractor.get_layer("MobileNetV3Small")
    rescaling_layers = [layer for layer in backbone.layers
                        if isinstance(layer, tf.keras.layers.Rescaling)]
    if not any(np.isclose(float(layer.scale), 1.0 / 127.5)
               and np.isclose(float(layer.offset), -1.0)
               for layer in rescaling_layers):
        raise ValueError("Candidate MobileNetV3 backbone is missing expected internal [0,255] to [-1,1] rescaling")

    return {
        "config": config,
        "classes": classes,
        "candidate_path": candidate_path,
        "production_path": production_path,
        "manifest_path": manifest_path,
        "test_rows": test_rows,
        "counts": counts,
        "output_dir": output_dir,
        "candidate_model": candidate_model,
        "production_model": production_model,
        "preflight": {
            "passed_before_test_image_access": True,
            "selected_checkpoint_matches_fine_tuning_report": True,
            "label_mapping_matches_saved_mapping_and_report": True,
            "input_preprocessing_contract_verified": input_contract,
            "candidate_input_output_shape": list(candidate_model.input_shape),
            "candidate_output_shape": list(candidate_model.output_shape),
            "production_keras_input_output_shape": list(production_model.input_shape),
            "production_keras_output_shape": list(production_model.output_shape),
            "candidate_softmax_output_verified": True,
            "candidate_internal_mobilenetv3_rescaling_verified": True,
            "expected_test_group_counts": counts,
            "supported_test_membership_matches_existing_grouped_manifest": len(candidate_supported_rows),
            "prior_training_validation_and_policy_selection_complete": True,
            "prior_full_validation_targets_met": False,
            "prior_policy_calibration_targets_met": False,
            "held_out_test_image_contents_read_before_preflight": False,
        },
    }


def metrics(rows: list[dict[str, str]], candidate_probabilities: np.ndarray,
            production_supported_probabilities: np.ndarray, classes: tuple[str, ...]) -> dict:
    if candidate_probabilities.shape != (len(rows), len(classes)):
        raise ValueError(f"Candidate output shape mismatch: {candidate_probabilities.shape}")
    supported_rows = [row for row in rows if row["triage_label"] == SUPPORTED]
    if production_supported_probabilities.shape != (len(supported_rows), 13):
        raise ValueError("Production baseline predictions are not aligned to held-out supported-food rows")
    if not np.isfinite(candidate_probabilities).all() or not np.isfinite(production_supported_probabilities).all():
        raise ValueError("Model predictions contain non-finite values")
    if (np.any(candidate_probabilities < 0) or np.any(candidate_probabilities > 1)
            or not np.allclose(candidate_probabilities.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError("Candidate outputs are not valid 14-way softmax probabilities")
    if (np.any(production_supported_probabilities < 0)
            or np.any(production_supported_probabilities > 1)
            or not np.allclose(production_supported_probabilities.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError("Production Keras outputs are not valid 13-way probabilities")

    predicted = candidate_probabilities.argmax(axis=1)
    accepted = predicted != 13
    truth = np.asarray([
        classes.index(row["food_class"]) if row["triage_label"] == SUPPORTED else -1
        for row in rows
    ], dtype=np.int64)
    supported_mask = np.asarray([row["triage_label"] == SUPPORTED for row in rows])
    nonfood_mask = np.asarray([row["triage_label"] == NON_FOOD for row in rows])
    unsupported_mask = np.asarray([row["triage_label"] == UNSUPPORTED for row in rows])
    supported_acceptance = float(accepted[supported_mask].mean())
    supported_top1 = float((predicted[supported_mask] == truth[supported_mask]).mean())
    effective_correct = float((accepted[supported_mask] &
                                (predicted[supported_mask] == truth[supported_mask])).mean())
    production_top1 = production_supported_probabilities.argmax(axis=1)
    production_accuracy = float((production_top1 == truth[supported_mask]).mean())
    nonfood_rejection = float((~accepted[nonfood_mask]).mean())
    unsupported_rejection = float((~accepted[unsupported_mask]).mean())

    per_class = {}
    for class_index, label in enumerate(classes[:13]):
        mask = supported_mask & (truth == class_index)
        if not mask.any():
            raise ValueError(f"No held-out supported examples for {label}")
        per_class[label] = {
            "samples": int(mask.sum()),
            "accepted_as_supported_food": int(accepted[mask].sum()),
            "rejected_as_non_food": int((~accepted[mask]).sum()),
            "acceptance": float(accepted[mask].mean()),
            "top1_correct": int((predicted[mask] == class_index).sum()),
            "top1_accuracy": float((predicted[mask] == class_index).mean()),
        }

    row_labels = [f"supported_food:{label}" for label in classes[:13]] + [NON_FOOD, UNSUPPORTED]
    col_labels = list(classes[:13]) + [NON_FOOD]
    confusion = np.zeros((15, 14), dtype=np.int64)
    for i, row in enumerate(rows):
        if row["triage_label"] == SUPPORTED:
            row_index = classes.index(row["food_class"])
        elif row["triage_label"] == NON_FOOD:
            row_index = 13
        else:
            row_index = 14
        confusion[row_index, int(predicted[i])] += 1

    policy_accuracy_drop = production_accuracy - effective_correct
    # Targets are reported from the frozen experiment configuration by the caller.
    return {
        "decision_policy": {"rule": FROZEN_POLICY, "confidence_threshold": None},
        "samples": {
            "supported_food": int(supported_mask.sum()),
            "non_food": int(nonfood_mask.sum()),
            "unsupported_food_evaluation_only": int(unsupported_mask.sum()),
        },
        "supported_food_acceptance_overall": supported_acceptance,
        "supported_food_acceptance_each_class_min": min(v["acceptance"] for v in per_class.values()),
        "supported_food_by_class": per_class,
        "non_food_rejection": nonfood_rejection,
        "unsupported_food_rejection": unsupported_rejection,
        "supported_food_top1_accuracy_14class": supported_top1,
        "effective_accepted_and_correct_accuracy": effective_correct,
        "production_13class_keras_top1_accuracy_same_supported_examples": production_accuracy,
        "classification_accuracy_drop_absolute_vs_keras": policy_accuracy_drop,
        "classification_accuracy_change_percentage_points_vs_keras": (effective_correct - production_accuracy) * 100.0,
        "confusion_matrix": {
            "rows_actual": row_labels,
            "columns_predicted": col_labels,
            "counts": confusion.tolist(),
        },
        "false_counts": {
            "supported_food_rejected_as_non_food": int((~accepted[supported_mask]).sum()),
            "supported_food_accepted_but_misclassified_as_another_food": int(
                (accepted[supported_mask] & (predicted[supported_mask] != truth[supported_mask])).sum()
            ),
            "non_food_false_acceptances_as_supported_food": int((accepted[nonfood_mask]).sum()),
            "non_food_correct_rejections": int((~accepted[nonfood_mask]).sum()),
            "unsupported_food_false_acceptances_as_supported_food": int((accepted[unsupported_mask]).sum()),
            "unsupported_food_rejections_as_non_food": int((~accepted[unsupported_mask]).sum()),
        },
        "per_class_production_baseline_accuracy": {
            label: float(np.mean([
                production_top1[i] == classes.index(label)
                for i, row in enumerate(supported_rows) if row["food_class"] == label
            ]))
            for label in classes[:13]
        },
    }


def acceptance_targets(config: dict, result: dict) -> dict:
    targets = config["acceptance_targets"]
    criteria = {
        "supported_food_acceptance_overall_min": result["supported_food_acceptance_overall"] >= targets["supported_food_acceptance_overall_min"],
        "supported_food_acceptance_each_class_min": result["supported_food_acceptance_each_class_min"] >= targets["supported_food_acceptance_each_class_min"],
        "non_food_rejection_min": result["non_food_rejection"] >= targets["non_food_rejection_min"],
        "unsupported_food_rejection_min": result["unsupported_food_rejection"] >= targets["unsupported_food_rejection_min"],
        "supported_classification_accuracy_max_drop_absolute": result["classification_accuracy_drop_absolute_vs_keras"] <= targets["supported_classification_accuracy_max_drop_absolute"],
    }
    return {
        "source": targets["source"],
        "targets": {key: value for key, value in targets.items() if key != "source"},
        "met": criteria,
        "all_met": all(criteria.values()),
        "class_acceptance_all_classes_at_least_min": all(
            value["acceptance"] >= targets["supported_food_acceptance_each_class_min"]
            for value in result["supported_food_by_class"].values()
        ),
        "classification_comparison_runtime": "14-class Keras candidate vs 13-class production Keras baseline on identical 2,719 supported-food test rows. The separately deployed TFLite 88.12% figure is not used in this Keras comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # This completes every gate that can be checked without opening test images.
    ready = preflight(args.config, args.output_dir)
    candidate_hash_before = _sha256(ready["candidate_path"])
    test_rows = ready["test_rows"]
    missing_paths = [row.get("path", "") for row in test_rows if not Path(row.get("path", "")).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} held-out image paths missing; first={missing_paths[0]}")

    started = time.perf_counter()
    candidate_probabilities = runner.predict_rows(ready["candidate_model"], test_rows)
    supported_rows = [row for row in test_rows if row["triage_label"] == SUPPORTED]
    production_probabilities = runner.predict_rows(ready["production_model"], supported_rows)
    elapsed = time.perf_counter() - started
    result = metrics(test_rows, candidate_probabilities, production_probabilities, ready["classes"])
    result["predeclared_targets"] = acceptance_targets(ready["config"], result)
    result["deployed_tflite_reference"] = {
        "accuracy": 0.8812063258550937,
        "source": str((ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_tflite_verification.json").resolve()),
        "comparison_to_keras_candidate_or_baseline": "Reported separately; not treated as an equivalent runtime comparison.",
    }

    if _sha256(ready["candidate_path"]) != candidate_hash_before:
        raise RuntimeError("Frozen candidate checkpoint changed during evaluation")
    report = {
        "experiment": "phase9_4_frozen_14class_heldout_evaluation",
        "status": "frozen_candidate_evaluated_once",
        "checkpoint": str(ready["candidate_path"].resolve()),
        "checkpoint_sha256": candidate_hash_before,
        "production_keras_baseline_checkpoint": str(ready["production_path"].resolve()),
        "manifest": str(ready["manifest_path"].resolve()),
        "label_mapping": {label: index for index, label in enumerate(ready["classes"])},
        "preflight": ready["preflight"],
        "policy": {"rule": FROZEN_POLICY, "thresholds": None},
        "evaluation_seconds": elapsed,
        "metrics": result,
        "limitations": [
            "The 14-class model rejects many non-food objects in the Open Images V7 test cohort.",
            "It is not a reliable unsupported-food detector; unsupported foods are not represented by the non_food training label.",
            "Unsupported foods can still receive high-confidence predictions as one of the 13 supported classes.",
            "Metrics reflect these held-out datasets and do not establish performance on every real-world image distribution.",
        ],
        "model_weights_modified": False,
        "test_evaluated_once": True,
        "threshold_optimization_performed": False,
        "tflite_export_performed": False,
    }
    ready["output_dir"].mkdir(parents=True, exist_ok=False)
    report_path = ready["output_dir"] / "heldout_evaluation_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "evaluation_seconds": elapsed,
        "metrics": result,
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
