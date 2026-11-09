"""Isolated float32 TFLite export, parity check, and frozen-cohort reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import tensorflow as tf

from train_phase9_3_14class import ROOT, preprocess_image

EXPERIMENTS = ROOT / "ml/experiments"
DEFAULT_CONFIG = EXPERIMENTS / "phase9_3_14class_config.json"
DEFAULT_OUTPUT = ROOT / "ml/artifacts/experiments/phase9_5_14class_tflite"
CHECKPOINT = ROOT / "ml/artifacts/experiments/phase9_3_14class_finetune/phase9_3_14class_finetune_best.keras"
FINE_REPORT = CHECKPOINT.parent / "fine_tune_report.json"
HELDOUT_REPORT = ROOT / "ml/artifacts/experiments/phase9_4_14class_heldout_evaluation/heldout_evaluation_report.json"
SUPPORTED = "supported_food"
NON_FOOD = "non_food"
UNSUPPORTED = "unsupported_food"
PARITY_SEED = "phase9_5_float32_val_parity_v1"
PARITY_MIN_ARGMAX_AGREEMENT = 0.999
PARITY_MAX_ABS_ERROR = 1e-4
PARITY_MEAN_ABS_ERROR = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _model_metrics(rows: list[dict[str, str]], probabilities: np.ndarray,
                   classes: tuple[str, ...]) -> dict:
    if probabilities.shape != (len(rows), len(classes)):
        raise ValueError(f"Unexpected probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Probability rows do not sum to one")
    predicted = probabilities.argmax(axis=1)
    groups = np.asarray([row["triage_label"] for row in rows])
    supported = groups == SUPPORTED
    non_food = groups == NON_FOOD
    unsupported = groups == UNSUPPORTED
    truth = np.asarray([
        classes.index(row["food_class"]) if row["triage_label"] == SUPPORTED else -1
        for row in rows
    ])
    accepted = predicted != 13
    per_class = {}
    for i, label in enumerate(classes[:13]):
        mask = supported & (truth == i)
        per_class[label] = {
            "samples": int(mask.sum()),
            "acceptance": float(accepted[mask].mean()) if mask.any() else None,
            "top1_accuracy": float((predicted[mask] == i).mean()) if mask.any() else None,
        }
    return {
        "samples": {
            "supported_food": int(supported.sum()),
            "non_food": int(non_food.sum()),
            "unsupported_food_evaluation_only": int(unsupported.sum()),
        },
        "supported_food_acceptance": float(accepted[supported].mean()),
        "supported_food_acceptance_by_class": per_class,
        "non_food_rejection": float((~accepted[non_food]).mean()),
        "unsupported_food_rejection": float((~accepted[unsupported]).mean()),
        "supported_food_top1_accuracy": float((predicted[supported] == truth[supported]).mean()),
        "effective_accepted_and_correct_accuracy": float(
            (accepted[supported] & (predicted[supported] == truth[supported])).mean()
        ),
        "false_counts": {
            "supported_food_rejected": int((~accepted[supported]).sum()),
            "non_food_false_acceptances": int(accepted[non_food].sum()),
            "unsupported_food_false_acceptances": int(accepted[unsupported].sum()),
        },
    }


def deterministic_validation_sample(rows: list[dict[str, str]], classes: tuple[str, ...],
                                     per_supported_class: int = 20,
                                     per_triage_group: int = 100) -> list[dict[str, str]]:
    """Select a fixed, label-stratified validation sample by stable source ID hash."""
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("Parity sample selection accepts validation rows only")
    selected = []
    for label in classes[:13]:
        group = [row for row in rows if row["triage_label"] == SUPPORTED and row["food_class"] == label]
        group.sort(key=lambda row: hashlib.sha256(
            f"{PARITY_SEED}:{row['source_dataset']}:{row['source_id']}".encode()
        ).digest())
        if len(group) < per_supported_class:
            raise ValueError(f"Not enough validation rows for {label}: {len(group)}")
        selected.extend(group[:per_supported_class])
    for triage in (NON_FOOD, UNSUPPORTED):
        group = [row for row in rows if row["triage_label"] == triage]
        group.sort(key=lambda row: hashlib.sha256(
            f"{PARITY_SEED}:{row['source_dataset']}:{row['source_id']}".encode()
        ).digest())
        if len(group) < per_triage_group:
            raise ValueError(f"Not enough validation rows for {triage}: {len(group)}")
        selected.extend(group[:per_triage_group])
    return selected


def preflight(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classes = tuple(config["classes"])
    mapping = config["class_to_idx"]
    if len(classes) != 14 or classes[-1] != NON_FOOD:
        raise ValueError("Expected the frozen 13-food plus non_food class order")
    if mapping != {name: index for index, name in enumerate(classes)}:
        raise ValueError("Config labels do not match the explicit frozen class indices")
    required = (CHECKPOINT, FINE_REPORT, HELDOUT_REPORT,
                ROOT / config["manifest"], ROOT / config["production_tflite_contract"],
                ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2.tflite",
                ROOT / "ml/mobile/mobilenet_v3_tflite_contract_v2.json",
                ROOT / "ml/mobile/mobilenet_v3_labels.txt")
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"Missing export input(s): {missing}")
    output_dir = output_dir.expanduser().resolve()
    artifact_root = (ROOT / "ml/artifacts/experiments").resolve()
    try:
        output_dir.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(f"New output directory must be under {artifact_root}") from error
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {output_dir}")

    fine = json.loads(FINE_REPORT.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_REPORT.read_text(encoding="utf-8"))
    mapping_artifact = json.loads((CHECKPOINT.parent / "class_mapping.json").read_text(encoding="utf-8"))
    prod_contract_path = ROOT / "ml/mobile/mobilenet_v3_tflite_contract_v2.json"
    prod_contract = json.loads(prod_contract_path.read_text(encoding="utf-8"))
    if Path(fine.get("selected_checkpoint", "")).resolve() != CHECKPOINT.resolve():
        raise ValueError("Fine-tuning report selected a different checkpoint")
    if fine.get("held_out_test", {}).get("evaluated") is not False:
        raise ValueError("Fine-tuning report does not record the test as untouched")
    if fine.get("classes") != list(classes) or fine.get("class_to_idx") != mapping:
        raise ValueError("Fine-tuning report class mapping does not match frozen policy")
    if mapping_artifact.get("classes") != list(classes) or mapping_artifact.get("class_to_idx") != mapping:
        raise ValueError("Saved checkpoint class mapping does not match frozen policy")
    if Path(heldout.get("checkpoint", "")).resolve() != CHECKPOINT.resolve():
        raise ValueError("Phase 9.4 held-out report used a different checkpoint")
    if heldout.get("checkpoint_sha256") != sha256(CHECKPOINT):
        raise ValueError("Selected checkpoint hash differs from the Phase 9.4 frozen evaluation artifact")
    if heldout.get("policy", {}).get("thresholds") is not None:
        raise ValueError("Phase 9.4 policy was not the frozen no-threshold argmax policy")
    if heldout.get("test_evaluated_once") is not True:
        raise ValueError("Frozen Keras held-out evaluation is not recorded as completed")
    if tuple(prod_contract.get("labels", [])) != classes[:13]:
        raise ValueError("Production TFLite class order does not match the first 13 candidate labels")
    expected_input = prod_contract.get("input", {})
    if (expected_input.get("tensor_shape") != [1, 224, 224, 3]
            or expected_input.get("dtype") != "float32"
            or expected_input.get("pixel_range") != [0, 255]
            or "shortest edge to 256" not in expected_input.get("resize", "")
            or "center crop 224x224" not in expected_input.get("resize", "")):
        raise ValueError("Existing TFLite preprocessing contract is not the expected raw-RGB input")

    manifest = ROOT / config["manifest"]
    all_rows = read_rows(manifest)
    validation_rows = [row for row in all_rows if row.get("split") == "validation"
                       and row.get("duplicate_label_conflict", "false").lower() != "true"]
    test_rows = [row for row in all_rows if row.get("split") == "test"
                 and row.get("duplicate_label_conflict", "false").lower() != "true"]
    if len(validation_rows) != 8456:
        raise ValueError(f"Unexpected validation row count {len(validation_rows)}")
    counts = {
        "supported_food": sum(row["triage_label"] == SUPPORTED for row in test_rows),
        "non_food": sum(row["triage_label"] == NON_FOOD for row in test_rows),
        "unsupported_food": sum(row["triage_label"] == UNSUPPORTED for row in test_rows),
    }
    if counts != {"supported_food": 2719, "non_food": 529, "unsupported_food": 22750}:
        raise ValueError(f"Held-out cohort count mismatch: {counts}")
    val_rows = [row for row in validation_rows if row["triage_label"] in (SUPPORTED, NON_FOOD, UNSUPPORTED)]
    sample = deterministic_validation_sample(val_rows, classes)
    if len(sample) != 460:
        raise ValueError(f"Expected 460 deterministic parity examples, found {len(sample)}")
    for row in validation_rows:
        if row["triage_label"] not in (SUPPORTED, NON_FOOD, UNSUPPORTED):
            raise ValueError("Unexpected validation category")
    return {
        "config": config, "classes": classes, "mapping": mapping,
        "manifest": manifest, "validation_rows": validation_rows,
        "parity_rows": sample, "test_rows": test_rows, "test_counts": counts,
        "output_dir": output_dir, "production_contract": prod_contract,
        "preflight": {
            "selected_checkpoint_matches_fine_tune_and_phase9_4_reports": True,
            "frozen_label_order_verified": True,
            "frozen_argmax_no_threshold_verified": True,
            "production_contract_and_preprocessing_verified": True,
            "validation_rows": len(validation_rows),
            "heldout_test_counts": counts,
            "test_evaluation_already_completed_for_keras_candidate": True,
        },
    }


def _lite_predict(interpreter: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape == (224, 224, 3):
        arr = arr[None, ...]
    if arr.shape != (1, 224, 224, 3) or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 255:
        raise ValueError("TFLite requires one finite RGB image in float32 [0,255]")
    interpreter.set_tensor(in_detail["index"], arr)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(out_detail["index"])[0].astype(np.float32)
    if probabilities.shape != (14,) or not np.isfinite(probabilities).all():
        raise ValueError("TFLite output has invalid shape or non-finite values")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError("TFLite output is not a 14-class probability distribution")
    return probabilities


def _preprocessed_batch(rows: list[dict[str, str]]) -> np.ndarray:
    return np.stack([preprocess_image(row["path"]) for row in rows]).astype(np.float32)


def _validate_interpreter(interpreter: tf.lite.Interpreter, classes: tuple[str, ...]) -> dict:
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"].tolist()) != (1, 224, 224, 3) or input_detail["dtype"] != np.float32:
        raise ValueError(f"TFLite input contract mismatch: {input_detail}")
    if tuple(output_detail["shape"].tolist()) != (1, len(classes)) or output_detail["dtype"] != np.float32:
        raise ValueError(f"TFLite output contract mismatch: {output_detail}")
    return {"input_shape": input_detail["shape"].tolist(), "input_dtype": str(input_detail["dtype"]),
            "output_shape": output_detail["shape"].tolist(), "output_dtype": str(output_detail["dtype"])}


def _contract(output_path: Path, classes: tuple[str, ...]) -> dict:
    return {
        "contract_version": 1,
        "model_file": output_path.relative_to(ROOT).as_posix(),
        "model_type": "TensorFlow Lite MobileNetV3 Small, 14-class image classifier candidate",
        "architecture": "mobilenet_v3_small",
        "input": {
            "width": 224, "height": 224, "color": "RGB", "dtype": "float32",
            "tensor_shape": [1, 224, 224, 3], "layout": "NHWC", "pixel_range": [0, 255],
            "resize": "Resize shortest edge to 256 pixels with bilinear interpolation, then center crop 224x224",
            "normalization": "TFLite graph preserves MobileNetV3 rescaling from [0,255] to [-1,1]; callers must not apply external normalization.",
        },
        "output": {
            "tensor_shape": [1, 14], "dtype": "float32",
            "interpretation": "14 softmax probabilities in labels order; outputs sum approximately to 1",
            "decision_policy": "argmax; indices 0-12 are supported food, index 13 is non_food; no additional confidence threshold",
        },
        "labels_file": (output_path.parent / "labels_14class.txt").relative_to(ROOT).as_posix(),
        "labels": list(classes),
        "class_to_idx": {label: index for index, label in enumerate(classes)},
        "nutrition_food_id": "only indices 0-12 map to the existing supported food IDs; non_food is a rejection label, not a nutrition ID",
    }


def export_and_verify(config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    ready = preflight(Path(config_path), Path(output_dir))
    checkpoint_hash = sha256(CHECKPOINT)
    production_paths = [ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2.tflite",
                        ROOT / "ml/mobile/mobilenet_v3_tflite_contract_v2.json",
                        ROOT / "ml/mobile/mobilenet_v3_labels.txt"]
    production_hashes_before = {str(path): sha256(path) for path in production_paths}
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    if tuple(model.input_shape) != (None, 224, 224, 3) or tuple(model.output_shape) != (None, 14):
        raise ValueError(f"Selected Keras checkpoint shape mismatch: {model.input_shape}, {model.output_shape}")
    head = model.get_layer("phase9_3_class_probabilities")
    if int(head.units) != 14 or head.activation.__name__ != "softmax":
        raise ValueError("Selected checkpoint does not contain the expected 14-output softmax head")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # No optimization or quantization: keep float32 input/output and weights.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flatbuffer = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=flatbuffer, num_threads=1)
    tensor_contract = _validate_interpreter(interpreter, ready["classes"])

    parity_rows = ready["parity_rows"]
    parity_images = _preprocessed_batch(parity_rows)
    keras_probabilities = np.asarray(model.predict(parity_images, batch_size=32, verbose=0), dtype=np.float32)
    lite_probabilities = np.stack([_lite_predict(interpreter, image) for image in parity_images])
    differences = np.abs(keras_probabilities - lite_probabilities)
    agreements = keras_probabilities.argmax(axis=1) == lite_probabilities.argmax(axis=1)
    parity = {
        "sample_count": len(parity_rows),
        "sampling": {
            "method": f"first deterministic SHA256 rank using {PARITY_SEED}; 20 per supported class, 100 non_food, 100 unsupported_food",
            "supported_food": sum(row["triage_label"] == SUPPORTED for row in parity_rows),
            "non_food": sum(row["triage_label"] == NON_FOOD for row in parity_rows),
            "unsupported_food": sum(row["triage_label"] == UNSUPPORTED for row in parity_rows),
        },
        "argmax_agreement": float(agreements.mean()),
        "argmax_agreement_count": int(agreements.sum()),
        "max_absolute_probability_difference": float(differences.max()),
        "mean_absolute_probability_difference": float(differences.mean()),
        "predeclared_parity_criteria": {
            "argmax_agreement_min": PARITY_MIN_ARGMAX_AGREEMENT,
            "max_abs_probability_difference_max": PARITY_MAX_ABS_ERROR,
            "mean_abs_probability_difference_max": PARITY_MEAN_ABS_ERROR,
        },
        "passed": bool(agreements.mean() >= PARITY_MIN_ARGMAX_AGREEMENT
                        and differences.max() <= PARITY_MAX_ABS_ERROR
                        and differences.mean() <= PARITY_MEAN_ABS_ERROR),
        "keras_metrics": _model_metrics(parity_rows, keras_probabilities, ready["classes"]),
        "tflite_metrics": _model_metrics(parity_rows, lite_probabilities, ready["classes"]),
    }
    if not parity["passed"]:
        raise RuntimeError(f"TFLite/Keras validation parity failed; held-out evaluation skipped: {parity}")

    # Only after conversion and validation parity pass, publish files within the
    # new isolated experiment directory. Production paths are never referenced
    # as output targets.
    ready["output_dir"].mkdir(parents=True, exist_ok=False)
    model_path = ready["output_dir"] / "mobilenet_v3_small_phase9_5_14class.tflite"
    labels_path = ready["output_dir"] / "labels_14class.txt"
    contract_path = ready["output_dir"] / "mobilenet_v3_14class_tflite_contract.json"
    for path in (model_path, labels_path, contract_path):
        if path.exists():
            raise FileExistsError(path)
    model_path.write_bytes(flatbuffer)
    labels_path.write_text("\n".join(ready["classes"]) + "\n", encoding="utf-8")
    contract = _contract(model_path, ready["classes"])
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    # Reporting-only TFLite evaluation on the already-frozen test cohort.
    test_rows = ready["test_rows"]
    missing = [row.get("path", "") for row in test_rows if not Path(row.get("path", "")).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frozen held-out paths are unavailable; first={missing[0]}")
    started = time.perf_counter()
    test_probabilities = []
    inference_ms = []
    for row in test_rows:
        image = preprocess_image(row["path"])
        tick = time.perf_counter()
        test_probabilities.append(_lite_predict(interpreter, image))
        inference_ms.append((time.perf_counter() - tick) * 1000.0)
    evaluation_seconds = time.perf_counter() - started
    test_probabilities = np.asarray(test_probabilities, dtype=np.float32)
    heldout_metrics = _model_metrics(test_rows, test_probabilities, ready["classes"])

    # Stable single-image CPU latency after warmup, one interpreter thread.
    sample_image = parity_images[0]
    for _ in range(10):
        _lite_predict(interpreter, sample_image)
    latency_ms = []
    for _ in range(100):
        tick = time.perf_counter()
        _lite_predict(interpreter, sample_image)
        latency_ms.append((time.perf_counter() - tick) * 1000.0)
    latency = {
        "device": "current host CPU; TensorFlow Lite interpreter, num_threads=1",
        "warmup_runs": 10, "measured_runs": len(latency_ms),
        "mean_ms": float(np.mean(latency_ms)),
        "median_ms": float(np.median(latency_ms)),
        "p95_ms": float(np.percentile(latency_ms, 95)),
        "full_test_inference_seconds": evaluation_seconds,
        "per_image_mean_ms_during_test": float(np.mean(inference_ms)),
    }

    keras_test = json.loads(HELDOUT_REPORT.read_text(encoding="utf-8"))["metrics"]
    comparison = {
        "same_frozen_heldout_rows": True,
        "tflite_supported_food_acceptance_minus_keras_pp":
            (heldout_metrics["supported_food_acceptance"] - keras_test["supported_food_acceptance_overall"]) * 100,
        "tflite_non_food_rejection_minus_keras_pp":
            (heldout_metrics["non_food_rejection"] - keras_test["non_food_rejection"]) * 100,
        "tflite_unsupported_food_rejection_minus_keras_pp":
            (heldout_metrics["unsupported_food_rejection"] - keras_test["unsupported_food_rejection"]) * 100,
        "tflite_supported_top1_minus_keras_pp":
            (heldout_metrics["supported_food_top1_accuracy"] - keras_test["supported_food_top1_accuracy_14class"]) * 100,
        "keras_supported_top1_accuracy": keras_test["supported_food_top1_accuracy_14class"],
        "keras_supported_acceptance": keras_test["supported_food_acceptance_overall"],
        "keras_non_food_rejection": keras_test["non_food_rejection"],
        "keras_unsupported_food_rejection": keras_test["unsupported_food_rejection"],
    }
    if sha256(CHECKPOINT) != checkpoint_hash:
        raise RuntimeError("Selected Keras checkpoint hash changed during export/evaluation")
    production_hashes_after = {str(path): sha256(path) for path in production_paths}
    if production_hashes_before != production_hashes_after:
        raise RuntimeError("Production TFLite assets changed during isolated export")

    report = {
        "experiment": "phase9_5_14class_float32_tflite_export_and_parity",
        "status": "exported_parity_passed_heldout_tflite_reported",
        "source_checkpoint": str(CHECKPOINT.resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "tflite_model": str(model_path.resolve()),
        "tflite_model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "labels_file": str(labels_path.resolve()),
        "contract_file": str(contract_path.resolve()),
        "labels": list(ready["classes"]),
        "decision_policy": "argmax over 14 outputs; indices 0-12 supported foods, 13 non_food; no threshold",
        "tflite_tensor_contract": tensor_contract,
        "input_preprocessing": contract["input"],
        "internal_normalization": "Verified by preserved model graph and Keras/TFLite parity when both receive the same raw RGB [0,255] inputs; no external normalization was applied.",
        "validation_parity": parity,
        "heldout_tflite_reporting_only": {
            "samples": heldout_metrics["samples"],
            "metrics": heldout_metrics,
            "comparison_to_previously_recorded_keras": comparison,
            "test_inference_seconds": evaluation_seconds,
        },
        "cpu_latency": latency,
        "conversion_warnings": [str(w.message) for w in caught],
        "production_artifacts_unchanged": production_hashes_before == production_hashes_after,
        "training_or_threshold_tuning_performed": False,
        "flutter_modified": False,
    }
    report_path = ready["output_dir"] / "tflite_export_verification.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = export_and_verify(args.config, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "tflite_model": result["tflite_model"],
        "model_size_bytes": result["model_size_bytes"],
        "validation_parity": result["validation_parity"],
        "heldout_metrics": result["heldout_tflite_reporting_only"]["metrics"],
        "cpu_latency": result["cpu_latency"],
        "report": str(Path(result["tflite_model"]).parent / "tflite_export_verification.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
