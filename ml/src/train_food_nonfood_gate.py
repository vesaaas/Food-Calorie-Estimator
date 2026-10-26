"""Train one isolated binary FOOD/NON-FOOD MobileNetV3 Small gate.

The frozen test split is not loaded or scored unless the best validation
checkpoint has a validation-only threshold satisfying every declared target.
All outputs are experimental and the script refuses to overwrite them.
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
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
else:
    from dataset import PHASE5_CLASSES


IMAGE_SIZE = 224
CLASSES = ("FOOD", "NON-FOOD")
SUPPORTED = "supported_food"
UNSUPPORTED = "unsupported_food"
NON_FOOD = "non_food"
PARALLEL_MAPS = 4
PREFETCH_BATCHES = 2


def _load_split(manifest: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("duplicate_label_conflict", "false").lower() == "true":
                continue
            path = Path(row.get("path", "")).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} image: {path}")
            triage = row.get("triage_label")
            if triage in (SUPPORTED, UNSUPPORTED):
                row["binary_label"] = "FOOD"
                row["binary_index"] = 0
            elif triage == NON_FOOD:
                row["binary_label"] = "NON-FOOD"
                row["binary_index"] = 1
            else:
                raise ValueError(f"Unexpected triage label in {split}: {triage!r}")
            row["path"] = str(path.resolve())
            rows.append(row)
    if not rows:
        raise ValueError(f"The {split} split has no usable images")
    labels = {row["binary_label"] for row in rows}
    if labels != set(CLASSES):
        raise ValueError(f"{split} must contain both binary classes; found {sorted(labels)}")
    supported_classes = {row.get("food_class") for row in rows if row["triage_label"] == SUPPORTED}
    missing = set(PHASE5_CLASSES) - supported_classes
    if missing:
        raise ValueError(f"{split} lacks supported-food metadata for {sorted(missing)}")
    if not any(row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101" for row in rows):
        raise ValueError(f"{split} lacks unsupported Food-101 food examples")
    if not any(row["triage_label"] == NON_FOOD and row["source_dataset"] == "Open Images V7" for row in rows):
        raise ValueError(f"{split} lacks the validated Open Images non-food examples")
    return rows


def _decode(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    shape = tf.shape(image)
    short_side = tf.minimum(shape[0], shape[1])
    scale = tf.cast(IMAGE_SIZE, tf.float32) / tf.cast(short_side, tf.float32)
    resized_h = tf.maximum(IMAGE_SIZE, tf.cast(tf.round(tf.cast(shape[0], tf.float32) * scale), tf.int32))
    resized_w = tf.maximum(IMAGE_SIZE, tf.cast(tf.round(tf.cast(shape[1], tf.float32) * scale), tf.int32))
    image = tf.image.resize(image, (resized_h, resized_w), method="bilinear", antialias=False)
    top = (resized_h - IMAGE_SIZE) // 2
    left = (resized_w - IMAGE_SIZE) // 2
    image = tf.image.crop_to_bounding_box(image, top, left, IMAGE_SIZE, IMAGE_SIZE)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return tf.cast(image, tf.float32)


def _dataset(rows: list[dict[str, str]], *, batch_size: int, weights: np.ndarray | None = None,
             training: bool = False, seed: int = 42) -> tf.data.Dataset:
    paths = np.asarray([row["path"] for row in rows], dtype=np.str_)
    labels = np.asarray([int(row["binary_index"]) for row in rows], dtype=np.int32)
    if weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            dataset = dataset.shuffle(min(len(rows), 10_000), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda path, label: (_decode(path), label),
            num_parallel_calls=PARALLEL_MAPS, deterministic=True,
        )
    else:
        sample_weights = np.asarray(weights, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))
        if training:
            dataset = dataset.shuffle(min(len(rows), 10_000), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda path, label, weight: (_decode(path), label, weight),
            num_parallel_calls=PARALLEL_MAPS, deterministic=True,
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if weights is not None or training:
        dataset = dataset.batch(batch_size, drop_remainder=False)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(PREFETCH_BATCHES)


def _sample_weights(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, float]]:
    labels = np.asarray([int(row["binary_index"]) for row in rows], dtype=np.int32)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Cannot calculate inverse-frequency weights for an empty class")
    # Each binary class contributes the same total loss; no examples are duplicated.
    class_weights = len(rows) / (len(CLASSES) * counts)
    sample_weights = class_weights[labels].astype(np.float32)
    return sample_weights, {name: float(class_weights[index]) for index, name in enumerate(CLASSES)}


def _augmentation(inputs: tf.Tensor) -> tf.Tensor:
    x = tf.keras.layers.RandomFlip("horizontal", seed=41, name="aug_flip")(inputs)
    x = tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=42, name="aug_rotation")(x)
    x = tf.keras.layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect", seed=43, name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=44, name="aug_contrast")(x)
    return tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255), seed=45,
                                             name="aug_brightness")(x)


def _build_model():
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32,
                            name="image_rgb_0_255")
    x = _augmentation(inputs)
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet",
        include_preprocessing=True, pooling="avg",
    )
    x = backbone(x)
    x = tf.keras.layers.Dropout(0.25, name="food_gate_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="food_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv3_small_food_nonfood_gate")


def _predict_tflite(rows: list[dict[str, str]], interpreter: tf.lite.Interpreter,
                    batch_size: int) -> np.ndarray:
    input_info = interpreter.get_input_details()[0]
    input_index = input_info["index"]
    target_shape = [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
    if input_info["shape"].tolist() != target_shape:
        interpreter.resize_tensor_input(input_index, target_shape, strict=False)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        input_index = input_info["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    outputs = []
    for batch in _dataset(rows, batch_size=batch_size):
        images = batch[0].numpy() if isinstance(batch, (tuple, list)) else batch.numpy()
        actual = len(images)
        if actual < batch_size:
            padded = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            padded[:actual] = images
            images = padded
        interpreter.set_tensor(input_index, images.astype(input_info["dtype"], copy=False))
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_index)[:actual])
    return np.concatenate(outputs, axis=0)


def _rates(rows: list[dict[str, str]], food_probabilities: np.ndarray, threshold: float) -> dict:
    food_prediction = food_probabilities >= threshold
    supported_mask = np.asarray([row["triage_label"] == SUPPORTED for row in rows])
    unsupported_mask = np.asarray([
        row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101"
        for row in rows
    ])
    nonfood_mask = np.asarray([row["triage_label"] == NON_FOOD for row in rows])
    supported_rates = {}
    supported_counts = {}
    for name in PHASE5_CLASSES:
        mask = np.asarray([row.get("food_class") == name and row["triage_label"] == SUPPORTED
                           for row in rows])
        supported_counts[name] = int(mask.sum())
        supported_rates[name] = float(np.mean(food_prediction[mask])) if mask.any() else None
    food_mask = supported_mask | unsupported_mask
    return {
        "threshold": float(threshold),
        "supported_food_pass_rate": float(np.mean(food_prediction[supported_mask])) if supported_mask.any() else None,
        "supported_class_pass_rates": supported_rates,
        "supported_class_counts": supported_counts,
        "unsupported_food101_pass_rate": float(np.mean(food_prediction[unsupported_mask])) if unsupported_mask.any() else None,
        "non_food_rejection_rate": float(np.mean(~food_prediction[nonfood_mask])) if nonfood_mask.any() else None,
        "food_false_negative_rate": float(np.mean(~food_prediction[food_mask])) if food_mask.any() else None,
        "non_food_false_positive_rate": float(np.mean(food_prediction[nonfood_mask])) if nonfood_mask.any() else None,
        "inconclusive_rate": 0.0,
    }


def _meets(rates: dict) -> bool:
    per_class = rates["supported_class_pass_rates"]
    return (
        rates["supported_food_pass_rate"] is not None
        and rates["supported_food_pass_rate"] >= 0.98
        and all(value is not None and value >= 0.95 for value in per_class.values())
        and rates["unsupported_food101_pass_rate"] is not None
        and rates["unsupported_food101_pass_rate"] >= 0.95
        and rates["non_food_rejection_rate"] is not None
        and rates["non_food_rejection_rate"] >= 0.95
    )


def _calibrate_validation(rows: list[dict[str, str]], probabilities: np.ndarray) -> dict:
    food_prob = probabilities[:, 0]
    candidates = np.unique(np.concatenate(([0.0], food_prob)))
    candidates = np.unique(np.concatenate((candidates, np.nextafter(candidates, 1.0))))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
    feasible = []
    best_failed = None
    for threshold in candidates:
        rates = _rates(rows, food_prob, float(threshold))
        if _meets(rates):
            feasible.append(rates)
        score = min(
            rates["supported_food_pass_rate"] / 0.98,
            min(rates["supported_class_pass_rates"].values()) / 0.95,
            rates["unsupported_food101_pass_rate"] / 0.95,
            rates["non_food_rejection_rate"] / 0.95,
        )
        rates["validation_target_attainment_score"] = float(score)
        if best_failed is None or (score, rates["non_food_rejection_rate"], threshold) > (
            best_failed["validation_target_attainment_score"],
            best_failed["non_food_rejection_rate"], best_failed["threshold"],
        ):
            best_failed = rates
    if not feasible:
        return {"accepted_on_validation": False, "selected_threshold": None,
                "best_failed_validation_point": best_failed,
                "reason": "No validation FOOD threshold meets all four acceptance requirements"}
    selected = max(feasible, key=lambda item: item["threshold"])
    return {"accepted_on_validation": True, "selected_threshold": selected["threshold"],
            "operating_metrics": selected, "selection_data": "validation_only",
            "selection_rule": "highest FOOD threshold satisfying every frozen validation requirement"}


def _confusion(rows: list[dict[str, str]], food_probabilities: np.ndarray,
               threshold: float) -> dict:
    true_food = np.asarray([row["binary_label"] == "FOOD" for row in rows])
    pred_food = food_probabilities >= threshold
    matrix = np.zeros((2, 2), dtype=np.int64)
    true_index = (~true_food).astype(np.int64)
    predicted_index = (~pred_food).astype(np.int64)
    np.add.at(matrix, (true_index, predicted_index), 1)
    return {"labels": list(CLASSES), "rows_true_columns_predicted": matrix.tolist()}


def _probability_summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90))}


def _distribution_report(rows: list[dict[str, str]], probabilities: np.ndarray) -> dict:
    food_prob = probabilities[:, 0]
    masks = {
        "supported_food": np.asarray([row["triage_label"] == SUPPORTED for row in rows]),
        "unsupported_food101": np.asarray([
            row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101"
            for row in rows
        ]),
        "non_food": np.asarray([row["triage_label"] == NON_FOOD for row in rows]),
    }
    return {name: _probability_summary(food_prob[mask]) for name, mask in masks.items()}


def _tflite_latency(interpreter: tf.lite.Interpreter) -> dict:
    info = interpreter.get_input_details()[0]
    index = info["index"]
    one_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
    if info["shape"].tolist() != one_shape:
        interpreter.resize_tensor_input(index, one_shape, strict=False)
        interpreter.allocate_tensors()
        info = interpreter.get_input_details()[0]
    values = np.zeros(one_shape, dtype=info["dtype"])
    for _ in range(5):
        interpreter.set_tensor(info["index"], values)
        interpreter.invoke()
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        interpreter.set_tensor(info["index"], values)
        interpreter.invoke()
        timings.append((time.perf_counter() - started) * 1000)
    return {"cpu_batch_1_median_ms": float(np.median(timings)),
            "cpu_batch_1_p90_ms": float(np.percentile(timings, 90)),
            "method": "TensorFlow Lite CPU, one thread, zero input, 5 warmups + 30 runs"}


def run(*, manifest: Path, checkpoint: Path, report: Path, epochs: int = 8,
        batch_size: int = 24, learning_rate: float = 2e-5, seed: int = 42) -> dict:
    tflite_path = checkpoint.with_suffix(".tflite")
    history_path = checkpoint.with_name(f"{checkpoint.stem}.training.json")
    for path in (checkpoint, report, tflite_path, history_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite experimental output: {path}")
    if epochs != 8 or batch_size != 24 or learning_rate != 2e-5 or seed != 42:
        raise ValueError("This authorized experiment requires 8 epochs, batch size 24, LR 2e-5, seed 42")

    train_rows = _load_split(manifest, "train")
    validation_rows = _load_split(manifest, "validation")
    train_weights, class_weights = _sample_weights(train_rows)
    validation_labels = np.asarray([int(row["binary_index"]) for row in validation_rows])
    validation_counts = np.bincount(validation_labels, minlength=2)
    validation_sample_weights = len(validation_rows) / (2.0 * validation_counts[validation_labels])

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model = _build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = model.fit(
        _dataset(train_rows, batch_size=batch_size, weights=train_weights, training=True, seed=seed),
        validation_data=_dataset(validation_rows, batch_size=batch_size,
                                 weights=validation_sample_weights, training=False, seed=seed),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min", save_best_only=True,
            save_weights_only=False,
        )],
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    history_values = {key: [float(value) for value in values]
                      for key, values in history.history.items()}
    history_path.write_text(json.dumps({"training_duration_seconds": training_seconds,
                                        "history": history_values}, indent=2) + "\n",
                            encoding="utf-8")
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best = tf.keras.models.load_model(checkpoint, compile=False)

    # Convert before validation calibration so the selected threshold matches
    # the eventual on-device float TFLite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(best)
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    interpreter.allocate_tensors()

    validation_probabilities = _predict_tflite(validation_rows, interpreter, batch_size)
    validation_policy = _calibrate_validation(validation_rows, validation_probabilities)
    result = {
        "task": "binary_food_nonfood_gate",
        "classes": list(CLASSES),
        "manifest": str(manifest.resolve()),
        "counts": {
            "train": len(train_rows), "validation": len(validation_rows),
            "test": None if not validation_policy["accepted_on_validation"] else "evaluated_once_after_validation_freeze",
            "train_by_class": dict(zip(CLASSES, np.bincount(
                [row["binary_index"] for row in train_rows], minlength=2).astype(int).tolist())),
            "validation_by_class": dict(zip(CLASSES, validation_counts.astype(int).tolist())),
        },
        "subgroup_counts": {
            split: {
                "supported_food": sum(row["triage_label"] == SUPPORTED for row in rows),
                "unsupported_food101": sum(row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101" for row in rows),
                "non_food": sum(row["triage_label"] == NON_FOOD for row in rows),
            } for split, rows in (("train", train_rows), ("validation", validation_rows))
        },
        "class_weights_train_inverse_frequency": class_weights,
        "input_pipeline": {"image_size": IMAGE_SIZE, "parallel_decode_calls": PARALLEL_MAPS,
                           "prefetch_batches": PREFETCH_BATCHES, "deterministic": True},
        "best_epoch": best_epoch,
        "best_validation_weighted_loss": float(min(history.history["val_loss"])),
        "best_epoch_validation_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
        "history": history_values,
        "training_duration_seconds": float(training_seconds),
        "validation_policy": validation_policy,
        "tflite": {"path": str(tflite_path.resolve()), "size_bytes": len(tflite_bytes),
                   "input_shape": interpreter.get_input_details()[0]["shape"].tolist(),
                   "output_shape": interpreter.get_output_details()[0]["shape"].tolist()},
        "keras_model_size_bytes": checkpoint.stat().st_size,
        "augmentations": ["horizontal flip", "small rotation", "small zoom", "contrast", "brightness"],
        "final_test_evaluated": False,
    }

    if validation_policy["accepted_on_validation"]:
        threshold = validation_policy["selected_threshold"]
        test_rows = _load_split(manifest, "test")
        test_probabilities = _predict_tflite(test_rows, interpreter, batch_size)
        test_rates = _rates(test_rows, test_probabilities[:, 0], threshold)
        test_confusion = _confusion(test_rows, test_probabilities[:, 0], threshold)
        result["counts"]["test"] = len(test_rows)
        result["counts"]["test_by_class"] = dict(zip(
            CLASSES, np.bincount([row["binary_index"] for row in test_rows], minlength=2).astype(int).tolist()
        ))
        result["subgroup_counts"]["test"] = {
            "supported_food": sum(row["triage_label"] == SUPPORTED for row in test_rows),
            "unsupported_food101": sum(row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101" for row in test_rows),
            "non_food": sum(row["triage_label"] == NON_FOOD for row in test_rows),
        }
        result["held_out_test"] = {
            "threshold_frozen_from_validation": threshold,
            "operating_metrics": test_rates,
            "confusion_matrix": test_confusion,
            "food_probability_distributions": _distribution_report(test_rows, test_probabilities),
            "test_used_for_threshold_selection": False,
        }
        result["final_test_evaluated"] = True
        result["final_acceptance_passed"] = _meets(test_rates)
    result["tflite"]["latency"] = _tflite_latency(interpreter)
    result["recommendation"] = (
        "PROMOTE BINARY GATE FOR FLUTTER INTEGRATION"
        if result.get("final_acceptance_passed") else "REJECT BINARY GATE"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                     prefix=f".{report.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(report)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def finalize_existing(*, manifest: Path, checkpoint: Path, report: Path,
                      batch_size: int = 24) -> dict:
    """Finish validation/test reporting for a completed run after an eval-only error."""
    tflite_path = checkpoint.with_suffix(".tflite")
    history_path = checkpoint.with_name(f"{checkpoint.stem}.training.json")
    if not checkpoint.is_file() or not tflite_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("Completed checkpoint, TFLite export, and training history are required")
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report}")
    training_record = json.loads(history_path.read_text(encoding="utf-8"))
    history = training_record["history"]
    if len(history.get("val_loss", [])) != 8:
        raise ValueError("Training history must contain exactly 8 completed epochs")
    train_rows = _load_split(manifest, "train")
    validation_rows = _load_split(manifest, "validation")
    train_weights, class_weights = _sample_weights(train_rows)
    del train_weights
    validation_labels = np.asarray([int(row["binary_index"]) for row in validation_rows])
    validation_counts = np.bincount(validation_labels, minlength=2)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    validation_probabilities = _predict_tflite(validation_rows, interpreter, batch_size)
    validation_policy = _calibrate_validation(validation_rows, validation_probabilities)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    result = {
        "task": "binary_food_nonfood_gate", "classes": list(CLASSES),
        "manifest": str(manifest.resolve()),
        "counts": {
            "train": len(train_rows), "validation": len(validation_rows), "test": None,
            "train_by_class": dict(zip(CLASSES, np.bincount(
                [row["binary_index"] for row in train_rows], minlength=2).astype(int).tolist())),
            "validation_by_class": dict(zip(CLASSES, validation_counts.astype(int).tolist())),
        },
        "subgroup_counts": {
            split: {
                "supported_food": sum(row["triage_label"] == SUPPORTED for row in rows),
                "unsupported_food101": sum(row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101" for row in rows),
                "non_food": sum(row["triage_label"] == NON_FOOD for row in rows),
            } for split, rows in (("train", train_rows), ("validation", validation_rows))
        },
        "class_weights_train_inverse_frequency": class_weights,
        "input_pipeline": {"image_size": IMAGE_SIZE, "parallel_decode_calls": PARALLEL_MAPS,
                           "prefetch_batches": PREFETCH_BATCHES, "deterministic": True},
        "best_epoch": best_epoch,
        "best_validation_weighted_loss": float(min(history["val_loss"])),
        "best_epoch_validation_accuracy": float(history["val_accuracy"][best_epoch - 1]),
        "history": history,
        "training_duration_seconds": float(training_record["training_duration_seconds"]),
        "validation_policy": validation_policy,
        "tflite": {"path": str(tflite_path.resolve()), "size_bytes": tflite_path.stat().st_size,
                   "input_shape": interpreter.get_input_details()[0]["shape"].tolist(),
                   "output_shape": interpreter.get_output_details()[0]["shape"].tolist()},
        "keras_model_size_bytes": checkpoint.stat().st_size,
        "augmentations": ["horizontal flip", "small rotation", "small zoom", "contrast", "brightness"],
        "final_test_evaluated": False,
    }
    if validation_policy["accepted_on_validation"]:
        threshold = validation_policy["selected_threshold"]
        test_rows = _load_split(manifest, "test")
        test_probabilities = _predict_tflite(test_rows, interpreter, batch_size)
        test_rates = _rates(test_rows, test_probabilities[:, 0], threshold)
        result["counts"]["test"] = len(test_rows)
        result["counts"]["test_by_class"] = dict(zip(
            CLASSES, np.bincount([row["binary_index"] for row in test_rows], minlength=2).astype(int).tolist()
        ))
        result["subgroup_counts"]["test"] = {
            "supported_food": sum(row["triage_label"] == SUPPORTED for row in test_rows),
            "unsupported_food101": sum(row["triage_label"] == UNSUPPORTED and row["source_dataset"] == "Food-101" for row in test_rows),
            "non_food": sum(row["triage_label"] == NON_FOOD for row in test_rows),
        }
        result["held_out_test"] = {
            "threshold_frozen_from_validation": threshold,
            "operating_metrics": test_rates,
            "confusion_matrix": _confusion(test_rows, test_probabilities[:, 0], threshold),
            "food_probability_distributions": _distribution_report(test_rows, test_probabilities),
            "test_used_for_threshold_selection": False,
        }
        result["final_test_evaluated"] = True
        result["final_acceptance_passed"] = _meets(test_rates)
    result["tflite"]["latency"] = _tflite_latency(interpreter)
    result["recommendation"] = (
        "PROMOTE BINARY GATE FOR FLUTTER INTEGRATION"
        if result.get("final_acceptance_passed") else "REJECT BINARY GATE"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                     prefix=f".{report.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(report)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Resume validation/test reporting from an existing completed checkpoint")
    args = parser.parse_args()
    try:
        if args.finalize_existing:
            result = finalize_existing(manifest=args.manifest, checkpoint=args.checkpoint,
                                       report=args.report, batch_size=args.batch_size)
        else:
            result = run(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Binary FOOD/NON-FOOD experiment failed: {error}\n")
    print(json.dumps({
        key: result.get(key) for key in (
            "task", "counts", "best_epoch", "best_validation_weighted_loss",
            "best_epoch_validation_accuracy", "training_duration_seconds",
            "validation_policy", "final_test_evaluated", "held_out_test",
            "keras_model_size_bytes", "tflite", "recommendation",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
