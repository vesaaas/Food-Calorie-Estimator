"""Isolated Phase 9.3 14-class MobileNetV3 transfer-learning runner.

Training is opt-in with ``--run``. This runner reads train and validation rows
only; it does not read test image paths, evaluate a test set, or export TFLite.
All checkpoints and reports are written to a new experiment-only directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "ml/experiments/phase9_3_14class_config.json"
IMAGE_SIZE = 224
SUPPORTED = "supported_food"
UNSUPPORTED = "unsupported_food"
NON_FOOD = "non_food"
TRAIN_SPLITS = {"train", "validation"}
SEED = 42


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    classes = config["classes"]
    if len(classes) != 14 or len(set(classes)) != 14 or classes[-1] != NON_FOOD:
        raise ValueError("The experiment must define 13 supported labels followed by non_food")
    expected = {name: index for index, name in enumerate(classes)}
    if config.get("class_to_idx") != expected:
        raise ValueError("Configured 14-class indices do not match the label order")
    metadata_path = ROOT / config["initial_checkpoint_metadata"]
    contract_path = ROOT / config["production_tflite_contract"]
    if not metadata_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Checkpoint metadata or production TFLite contract is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if metadata.get("classes") != classes[:-1] or contract.get("labels") != classes[:-1]:
        raise ValueError("The 14-class mapping must preserve the existing first 13 class indices")
    return config


def partition_manifest_rows(
    rows: list[dict[str, str]], supported_classes: tuple[str, ...], *, require_complete: bool = True,
) -> dict[str, list[dict[str, str]]]:
    """Select only fit/validation cohorts; held-out test rows are discarded.

    Unsupported foods are retained only as a distinct validation rejection
    cohort. They are never mapped to the non_food training label.
    """
    result = {
        "train_fit": [],
        "validation_fit": [],
        "validation_supported": [],
        "validation_non_food": [],
        "validation_unsupported": [],
    }
    supported = set(supported_classes)
    for row in rows:
        split = row.get("split")
        if split not in TRAIN_SPLITS:
            # In particular, do not inspect the path of any held-out test row.
            continue
        if row.get("duplicate_label_conflict", "false").lower() == "true":
            continue
        triage = row.get("triage_label")
        if triage == SUPPORTED:
            if row.get("food_class") not in supported:
                raise ValueError(f"Unknown supported food class: {row.get('food_class')!r}")
            row = {**row, "label": row["food_class"]}
            result["validation_supported" if split == "validation" else "train_fit"].append(row)
            if split == "validation":
                result["validation_fit"].append(row)
        elif triage == NON_FOOD:
            if row.get("source_dataset") != "Open Images V7":
                raise ValueError("The non_food target must come from the verified Open Images subset")
            row = {**row, "label": NON_FOOD}
            result["validation_non_food" if split == "validation" else "train_fit"].append(row)
            if split == "validation":
                result["validation_fit"].append(row)
        elif triage == UNSUPPORTED and split == "validation":
            result["validation_unsupported"].append(row)
        elif triage == UNSUPPORTED:
            # Shared robustness manifests contain unrelated unsupported-food
            # training rows; this experiment intentionally ignores them.
            continue
        else:
            raise ValueError(f"Unexpected triage label in {split}: {triage!r}")

    if require_complete:
        for key in ("train_fit", "validation_fit", "validation_supported",
                    "validation_non_food", "validation_unsupported"):
            if not result[key]:
                raise ValueError(f"Required {key} partition is empty")
        for split_name, split_rows in (
            ("train", result["train_fit"]),
            ("validation", result["validation_supported"]),
        ):
            labels = {row["label"] for row in split_rows}
            missing = supported - labels
            if missing:
                raise ValueError(f"{split_name} is missing supported classes: {sorted(missing)}")
        if not any(row["label"] == NON_FOOD for row in result["train_fit"]):
            raise ValueError("Training data is missing non_food examples")
        if not any(row["label"] == NON_FOOD for row in result["validation_fit"]):
            raise ValueError("Validation data is missing non_food examples")
    return result


def load_fit_and_validation_rows(manifest_path: Path, supported_classes: tuple[str, ...]) -> dict:
    """Read and validate train/validation files without touching test images."""
    rows = []
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") not in TRAIN_SPLITS:
                continue
            if row.get("duplicate_label_conflict", "false").lower() == "true":
                continue
            triage = row.get("triage_label")
            split = row["split"]
            if triage == UNSUPPORTED and split == "train":
                continue
            if triage not in (SUPPORTED, NON_FOOD, UNSUPPORTED):
                raise ValueError(f"Unexpected triage label: {triage!r}")
            path = Path(row["path"]).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} image: {path}")
            row["path"] = str(path.resolve())
            rows.append(row)
    partitions = partition_manifest_rows(rows, supported_classes)
    verify_train_validation_isolation(
        partitions["train_fit"],
        partitions["validation_supported"]
        + partitions["validation_non_food"]
        + partitions["validation_unsupported"],
    )
    return partitions


def _hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def verify_train_validation_isolation(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]],
    *, near_duplicate_distance: int = 5,
) -> None:
    """Reject group, exact-hash, or dHash<=5 overlap between fit and validation."""
    for field in ("group_id", "sha256", "dhash"):
        train_values = {row.get(field, "").strip() for row in train_rows}
        validation_values = {row.get(field, "").strip() for row in validation_rows}
        train_values.discard("")
        validation_values.discard("")
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(f"Train/validation {field} overlap: {len(overlap)}")

    bands: dict[tuple[int, int], list[int]] = {}
    for row in train_rows:
        code = int(row["dhash"], 16)
        for band in range(8):
            bands.setdefault((band, (code >> (band * 8)) & 0xFF), []).append(code)
    for row in validation_rows:
        code = int(row["dhash"], 16)
        for band in range(8):
            key = (band, (code >> (band * 8)) & 0xFF)
            if any(_hamming(code, prior) <= near_duplicate_distance
                   for prior in bands.get(key, ())):
                raise ValueError("Train/validation dHash near-duplicate overlap")


def preprocess_image(path: str | Path) -> np.ndarray:
    """Production-compatible RGB float32 [0,255] NHWC preprocessing."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Image has empty dimensions: {path}")
        scale = 256 / min(width, height)
        resized = (int(width * scale), int(height * scale))
        image = image.resize(resized, Image.Resampling.BILINEAR)
        left = int(round((image.width - IMAGE_SIZE) / 2.0))
        top = int(round((image.height - IMAGE_SIZE) / 2.0))
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        pixels = np.asarray(image, dtype=np.float32)
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected preprocessed shape {pixels.shape} from {path}")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError(f"Preprocessor returned pixels outside finite [0,255]: {path}")
    return pixels


def _decode_path(path_tensor: tf.Tensor) -> np.ndarray:
    raw = path_tensor.numpy()
    path = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return preprocess_image(path)


def make_dataset(
    rows: list[dict[str, str]], class_to_idx: dict[str, int], *, batch_size: int,
    training: bool, seed: int,
) -> tf.data.Dataset:
    if not rows:
        raise ValueError("Cannot create a dataset from no rows")
    paths = np.asarray([row["path"] for row in rows], dtype=np.str_)
    labels = np.asarray([class_to_idx[row["label"]] for row in rows], dtype=np.int32)
    if training:
        counts = np.bincount(labels, minlength=len(class_to_idx)).astype(np.float64)
        if np.any(counts == 0):
            absent = [name for name, index in class_to_idx.items() if counts[index] == 0]
            raise ValueError(f"Training set is missing classes for balancing: {absent}")
        weights_per_class = len(rows) / (len(class_to_idx) * counts)
        sample_weights = weights_per_class[labels].astype(np.float32)
        components = (paths, labels, sample_weights)
    else:
        components = (paths, labels)

    dataset = tf.data.Dataset.from_tensor_slices(components)
    if training:
        dataset = dataset.shuffle(min(len(rows), 10_000), seed=seed,
                                  reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, label: tf.Tensor, *sample_weight: tf.Tensor):
        image = tf.py_function(_decode_path, [path], Tout=tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        if sample_weight:
            return image, label, sample_weight[0]
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=2, deterministic=True)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(1)


def _augmentation(inputs: tf.Tensor) -> tf.Tensor:
    x = tf.keras.layers.RandomFlip("horizontal", seed=41, name="phase9_3_aug_flip")(inputs)
    x = tf.keras.layers.RandomRotation(
        0.04, fill_mode="reflect", seed=42, name="phase9_3_aug_rotation",
    )(x)
    x = tf.keras.layers.RandomZoom(
        (-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect", seed=43,
        name="phase9_3_aug_zoom",
    )(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=44, name="phase9_3_aug_contrast")(x)
    return tf.keras.layers.RandomBrightness(
        0.12, value_range=(0, 255), seed=45, name="phase9_3_aug_brightness",
    )(x)


def build_14class_model(
    base_model: tf.keras.Model, classes: tuple[str, ...], *, augmentation: bool = True,
) -> tf.keras.Model:
    """Reuse the frozen 13-class checkpoint graph and replace only its Dense head."""
    if tuple(base_model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected checkpoint input shape: {base_model.input_shape}")
    if tuple(base_model.output_shape) != (None, 13):
        raise ValueError(f"Expected 13-way checkpoint output, got {base_model.output_shape}")
    if len(classes) != 14 or classes[-1] != NON_FOOD:
        raise ValueError("Expected 13 existing food labels followed by non_food")
    old_head = base_model.get_layer("class_probabilities")
    if int(old_head.units) != 13 or old_head.activation.__name__ != "softmax":
        raise ValueError("Checkpoint classification head is not the expected 13-way softmax")

    feature_extractor = tf.keras.Model(
        base_model.input, old_head.input, name="phase9_3_frozen_feature_extractor",
    )
    inputs = tf.keras.Input(
        (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image_rgb_0_255",
    )
    model_input = _augmentation(inputs) if augmentation else inputs
    features = feature_extractor(model_input)
    outputs = tf.keras.layers.Dense(
        len(classes), activation="softmax", name="phase9_3_class_probabilities",
    )(features)
    model = tf.keras.Model(inputs, outputs, name="phase9_3_mobilenet_v3_small_14class")
    freeze_backbone_train_head(model)
    if tuple(model.output_shape) != (None, 14):
        raise ValueError(f"Constructed model has unexpected output shape: {model.output_shape}")
    return model


def freeze_backbone_train_head(model: tf.keras.Model) -> None:
    extractor = model.get_layer("phase9_3_frozen_feature_extractor")
    extractor.trainable = False
    for layer in model.layers:
        layer.trainable = layer.name == "phase9_3_class_probabilities"


def unfreeze_last_backbone_fraction(
    model: tf.keras.Model, *, fraction: float = 0.20,
) -> list[str]:
    if not 0 < fraction <= 1:
        raise ValueError("Fine-tune fraction must be in (0,1]")
    extractor = model.get_layer("phase9_3_frozen_feature_extractor")
    backbone = extractor.get_layer("MobileNetV3Small")
    extractor.trainable = True
    for layer in extractor.layers:
        layer.trainable = False
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    candidates = [
        layer for layer in backbone.layers
        if not isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.BatchNormalization))
    ]
    count = math.floor(len(candidates) * fraction)
    if count < 1:
        raise ValueError(
            f"Backbone has too few eligible layers to unfreeze at most {fraction:.0%}"
        )
    selected = candidates[-count:]
    for layer in selected:
        layer.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    for layer in model.layers:
        layer.trainable = layer.name in (
            "phase9_3_class_probabilities", "phase9_3_frozen_feature_extractor",
        )
    return [layer.name for layer in selected]


def predict_rows(model: tf.keras.Model, rows: list[dict[str, str]], *, batch_size: int = 32) -> np.ndarray:
    if not rows:
        return np.empty((0, int(model.output_shape[-1])), dtype=np.float32)
    paths = tf.data.Dataset.from_tensor_slices(
        np.asarray([row["path"] for row in rows], dtype=np.str_),
    )

    def decode(path: tf.Tensor) -> tf.Tensor:
        image = tf.py_function(_decode_path, [path], Tout=tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image

    dataset = paths.map(decode, num_parallel_calls=2, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(1)
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)
    if probabilities.shape != (len(rows), int(model.output_shape[-1])):
        raise ValueError(f"Unexpected prediction shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite validation probabilities")
    return probabilities


def validation_metrics(
    rows: list[dict[str, str]], probabilities: np.ndarray,
    classes: tuple[str, ...], baseline_supported_accuracy: float,
    acceptance_targets: dict | None = None,
) -> dict:
    if len(rows) != len(probabilities) or probabilities.shape[1] != len(classes):
        raise ValueError("Validation rows and 14-way probability matrix are misaligned")
    if not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities contain non-finite values")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Validation probabilities fall outside [0,1]")

    class_to_idx = {name: index for index, name in enumerate(classes)}
    predictions = probabilities.argmax(axis=1)
    supported_indices = [i for i, row in enumerate(rows) if row["triage_label"] == SUPPORTED]
    nonfood_indices = [i for i, row in enumerate(rows) if row["triage_label"] == NON_FOOD]
    unsupported_indices = [i for i, row in enumerate(rows) if row["triage_label"] == UNSUPPORTED]
    if not supported_indices or not nonfood_indices or not unsupported_indices:
        raise ValueError("Validation must include supported, non_food, and unsupported_food groups")

    supported_accept = [predictions[i] != class_to_idx[NON_FOOD] for i in supported_indices]
    supported_correct = [
        predictions[i] == class_to_idx[rows[i]["food_class"]] for i in supported_indices
    ]
    per_class = {}
    for label in classes[:-1]:
        indices = [i for i in supported_indices if rows[i]["food_class"] == label]
        if not indices:
            raise ValueError(f"Validation has no supported examples for {label}")
        per_class[label] = {
            "samples": len(indices),
            "acceptance": float(np.mean(predictions[indices] != class_to_idx[NON_FOOD])),
            "top1_accuracy": float(np.mean(predictions[indices] == class_to_idx[label])),
        }
    nonfood_rejection = float(np.mean(predictions[nonfood_indices] == class_to_idx[NON_FOOD]))
    unsupported_rejection = float(np.mean(predictions[unsupported_indices] == class_to_idx[NON_FOOD]))
    candidate_accuracy = float(np.mean(supported_correct))
    accuracy_drop = float(baseline_supported_accuracy - candidate_accuracy)
    configured_targets = acceptance_targets or {
        "supported_food_acceptance_overall_min": 0.97,
        "supported_food_acceptance_each_class_min": 0.95,
        "non_food_rejection_min": 0.80,
        "unsupported_food_rejection_min": 0.50,
        "supported_classification_accuracy_max_drop_absolute": 0.02,
    }
    targets = {
        "supported_food_acceptance_overall_min": configured_targets[
            "supported_food_acceptance_overall_min"
        ],
        "supported_food_acceptance_each_class_min": configured_targets[
            "supported_food_acceptance_each_class_min"
        ],
        "non_food_rejection_min": configured_targets["non_food_rejection_min"],
        "unsupported_food_rejection_min": configured_targets["unsupported_food_rejection_min"],
        "supported_classification_accuracy_max_drop_absolute": configured_targets[
            "supported_classification_accuracy_max_drop_absolute"
        ],
    }
    overall_acceptance = float(np.mean(supported_accept))
    gate = {
        "supported_food_acceptance_overall": overall_acceptance,
        "supported_food_acceptance_each_class_min": min(
            result["acceptance"] for result in per_class.values()
        ),
        "non_food_rejection": nonfood_rejection,
        "unsupported_food_rejection": unsupported_rejection,
        "supported_classification_accuracy": candidate_accuracy,
        "production_13class_validation_accuracy": float(baseline_supported_accuracy),
        "supported_classification_accuracy_drop_absolute": accuracy_drop,
    }
    gate["meets_targets"] = bool(
        overall_acceptance >= targets["supported_food_acceptance_overall_min"]
        and gate["supported_food_acceptance_each_class_min"] >= targets["supported_food_acceptance_each_class_min"]
        and nonfood_rejection >= targets["non_food_rejection_min"]
        and unsupported_rejection >= targets["unsupported_food_rejection_min"]
        and accuracy_drop <= targets["supported_classification_accuracy_max_drop_absolute"]
    )
    return {
        "decision_policy": {
            "rule": "argmax over the 14 outputs; accept as supported food iff predicted index is 0..12; reject iff predicted index is 13",
            "confidence_threshold": None,
            "validation_gate_required_before_candidate_acceptance": True,
        },
        "targets": targets,
        "acceptance_and_rejection": gate,
        "supported_food_by_class": per_class,
        "samples": {
            "supported_food": len(supported_indices),
            "non_food": len(nonfood_indices),
            "unsupported_food_evaluation_only": len(unsupported_indices),
        },
        "classification_accuracy_is_distinct_from_acceptance": True,
    }


def fit_stage(
    model: tf.keras.Model, train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset, *, checkpoint_path: Path,
    history_path: Path, epochs: int, learning_rate: float, patience: int = 2,
) -> dict:
    """Fit one explicitly bounded stage and save its best val-loss checkpoint."""
    if epochs < 1 or learning_rate <= 0 or patience < 0:
        raise ValueError("Invalid stage training bounds")
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint or training log")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path), monitor="val_loss", mode="min",
                save_best_only=True, save_weights_only=False, verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience,
                restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.CSVLogger(str(history_path)),
        ],
        verbose=2,
        shuffle=False,
    )
    duration = time.perf_counter() - started
    if not checkpoint_path.is_file():
        raise RuntimeError("ModelCheckpoint did not create the expected best checkpoint")
    values = {key: [float(value) for value in series] for key, series in history.history.items()}
    if not values.get("val_loss"):
        raise RuntimeError("Training completed without validation-loss history")
    return {
        "epochs_completed": len(values["val_loss"]),
        "best_epoch": int(np.argmin(values["val_loss"])) + 1,
        "best_validation_loss": float(np.min(values["val_loss"])),
        "training_seconds": float(duration),
        "history": values,
        "checkpoint": str(checkpoint_path.resolve()),
        "history_csv": str(history_path.resolve()),
    }


def _set_seeds(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _atomic_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment report: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_config_paths(config: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    manifest = ROOT / config["manifest"]
    checkpoint = ROOT / config["initial_checkpoint"]
    expected_root = (ROOT / "ml/artifacts/experiments").resolve()
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"Output must remain under isolated experiment root {expected_root}") from error
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output_dir}")
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Manifest or initial checkpoint is missing")
    return manifest, checkpoint, output_dir


def resolve_finetune_resume(
    config: dict, *, resume_checkpoint: Path | None = None,
    head_report_path: Path | None = None, output_dir: Path | None = None,
    experiment_root: Path | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Validate a fine-tune-only resume without loading images or writing files."""
    experiment_root = (experiment_root or (ROOT / "ml/artifacts/experiments")).resolve()
    head_dir = (ROOT / config["output_directory"]).resolve()
    resume_checkpoint = Path(resume_checkpoint or head_dir / "phase9_3_14class_head_best.keras").resolve()
    head_report_path = Path(head_report_path or head_dir / "training_report.json").resolve()
    output_dir = Path(output_dir or experiment_root / "phase9_3_14class_finetune").resolve()
    try:
        output_dir.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError(f"Fine-tune outputs must stay under {experiment_root}") from error
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fine-tune experiment directory: {output_dir}")
    if not resume_checkpoint.is_file() or not head_report_path.is_file():
        raise FileNotFoundError("The head-only checkpoint and its report are both required")
    if resume_checkpoint == (ROOT / config["initial_checkpoint"]).resolve():
        raise ValueError("Fine-tune-only must resume the 14-class head checkpoint, not production")
    if output_dir == resume_checkpoint.parent or output_dir == head_report_path.parent:
        raise ValueError("Fine-tune outputs must be separate from the head-only experiment")
    try:
        output_dir.relative_to(head_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Fine-tune output directory cannot be inside the head-only experiment")

    head_report = json.loads(head_report_path.read_text(encoding="utf-8"))
    expected_checkpoint = Path(head_report.get("selected_checkpoint", "")).resolve()
    stage_checkpoint = Path(
        head_report.get("stages", {}).get("frozen_head", {}).get("checkpoint", "")
    ).resolve()
    if resume_checkpoint not in (expected_checkpoint, stage_checkpoint):
        raise ValueError("Resume checkpoint is not the checkpoint recorded by the head-only report")
    if head_report.get("classes") != config["classes"]:
        raise ValueError("Head-only report class order differs from the configured 14-class order")
    if head_report.get("class_to_idx") != config["class_to_idx"]:
        raise ValueError("Head-only report class indices differ from the configured mapping")
    if head_report.get("experiment") != config["experiment"]:
        raise ValueError("Head-only report belongs to a different experiment")
    head_metrics = head_report.get("stages", {}).get("frozen_head", {}).get("validation")
    if not isinstance(head_metrics, dict) or not isinstance(
        head_metrics.get("acceptance_and_rejection"), dict,
    ):
        raise ValueError("Head-only report is missing validation metrics")
    if head_report.get("held_out_test", {}).get("evaluated") is not False:
        raise ValueError("Head-only report does not confirm that test evaluation was skipped")
    return resume_checkpoint, head_report_path, output_dir, head_report


def _metric_comparison(head: dict, fine: dict) -> dict:
    head_gate = head["acceptance_and_rejection"]
    fine_gate = fine["acceptance_and_rejection"]
    fields = (
        "supported_food_acceptance_overall",
        "supported_food_acceptance_each_class_min",
        "non_food_rejection",
        "unsupported_food_rejection",
        "supported_classification_accuracy",
        "production_13class_validation_accuracy",
        "supported_classification_accuracy_drop_absolute",
    )
    per_class = {
        name: {
            "head_only_acceptance": head["supported_food_by_class"][name]["acceptance"],
            "fine_tuned_acceptance": fine["supported_food_by_class"][name]["acceptance"],
            "delta": fine["supported_food_by_class"][name]["acceptance"]
            - head["supported_food_by_class"][name]["acceptance"],
        }
        for name in head["supported_food_by_class"]
    }
    return {
        "summary": {
            field: {
                "head_only": head_gate[field],
                "fine_tuned": fine_gate[field],
                "delta": fine_gate[field] - head_gate[field],
            }
            for field in fields
        },
        "supported_food_acceptance_by_class": per_class,
    }


def prepare_finetune_model(checkpoint: Path, classes: tuple[str, ...]) -> tuple[tf.keras.Model, list[str], dict]:
    """Load the head checkpoint and verify the bounded trainable layer set in memory."""
    model = tf.keras.models.load_model(checkpoint, compile=False)
    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected resumed model input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, len(classes)):
        raise ValueError(f"Resumed checkpoint is not the configured 14-way model: {model.output_shape}")
    head_layer = model.get_layer("phase9_3_class_probabilities")
    if int(head_layer.units) != 14 or head_layer.activation.__name__ != "softmax":
        raise ValueError("Resumed checkpoint does not contain the expected 14-way softmax head")

    fine_layers = unfreeze_last_backbone_fraction(model, fraction=0.20)
    backbone = model.get_layer("phase9_3_frozen_feature_extractor").get_layer("MobileNetV3Small")
    batch_norm_layers = [
        layer for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    batch_norm_frozen = all(not layer.trainable for layer in batch_norm_layers)
    candidates = [
        layer for layer in backbone.layers
        if not isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.BatchNormalization))
    ]
    expected_count = math.floor(len(candidates) * 0.20)
    if expected_count < 1:
        raise ValueError("Backbone has too few eligible layers for a <=20 percent fine-tune")
    actual_trainable = {
        layer.name for layer in candidates if layer.trainable
    }
    if not batch_norm_frozen:
        raise RuntimeError("Fine-tune setup unexpectedly enabled a BatchNormalization layer")
    if actual_trainable != set(fine_layers) or len(actual_trainable) != expected_count:
        raise RuntimeError("Fine-tune setup enabled layers outside the configured last 20 percent")
    if not head_layer.trainable:
        raise RuntimeError("The 14-class output head must remain trainable during fine-tuning")
    return model, fine_layers, {
        "batch_normalization_layer_count": len(batch_norm_layers),
        "batch_normalization_frozen": batch_norm_frozen,
        "eligible_backbone_layer_count": len(candidates),
        "trainable_backbone_layer_count": len(actual_trainable),
        "trainable_head": bool(head_layer.trainable),
    }


def run_finetune_only(
    *, config_path: Path = DEFAULT_CONFIG,
    resume_checkpoint: Path | None = None,
    head_report_path: Path | None = None,
    output_dir: Path | None = None,
    max_epochs: int = 2,
    seed: int = SEED,
) -> dict:
    """Resume from the best 14-class head checkpoint and run only fine-tuning."""
    config = load_config(config_path)
    resume_checkpoint, head_report_path, output_dir, head_report = resolve_finetune_resume(
        config,
        resume_checkpoint=resume_checkpoint,
        head_report_path=head_report_path,
        output_dir=output_dir,
    )
    if max_epochs < 1:
        raise ValueError("Fine-tuning epochs must be positive")
    max_epochs = min(max_epochs, 2)
    classes = tuple(config["classes"])
    manifest = ROOT / config["manifest"]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    partitions = load_fit_and_validation_rows(manifest, classes[:-1])
    _set_seeds(seed)

    checkpoint_hash_before = _sha256_file(resume_checkpoint)
    model, fine_layers, fine_tune_model_checks = prepare_finetune_model(resume_checkpoint, classes)
    batch_norm_frozen = fine_tune_model_checks["batch_normalization_frozen"]

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "class_mapping.json").write_text(
        json.dumps({"classes": list(classes), "class_to_idx": config["class_to_idx"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    train_dataset = make_dataset(
        partitions["train_fit"], config["class_to_idx"],
        batch_size=24, training=True, seed=seed,
    )
    validation_dataset = make_dataset(
        partitions["validation_fit"], config["class_to_idx"],
        batch_size=24, training=False, seed=seed,
    )
    fine_checkpoint = output_dir / "phase9_3_14class_finetune_best.keras"
    fine_history = output_dir / "phase9_3_14class_finetune_history.csv"
    fine_stage = fit_stage(
        model, train_dataset, validation_dataset,
        checkpoint_path=fine_checkpoint, history_path=fine_history,
        epochs=max_epochs, learning_rate=1e-5,
    )
    best_model = tf.keras.models.load_model(fine_checkpoint, compile=False)
    validation_rows = partitions["validation_fit"] + partitions["validation_unsupported"]
    fine_probabilities = predict_rows(best_model, validation_rows)
    baseline_accuracy = head_report["validation"]["acceptance_and_rejection"][
        "production_13class_validation_accuracy"
    ]
    fine_metrics = validation_metrics(
        validation_rows, fine_probabilities, classes, baseline_accuracy,
        config["acceptance_targets"],
    )
    head_metrics = head_report["validation"]
    improvement_loss = fine_stage["best_validation_loss"] < head_report["stages"]["frozen_head"][
        "best_validation_loss"
    ]
    selected_checkpoint = fine_checkpoint if improvement_loss else resume_checkpoint
    checkpoint_hash_after = _sha256_file(resume_checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("Head-only source checkpoint changed during fine-tuning")

    report = {
        "experiment": config["experiment"],
        "mode": "fine_tune_only_resume",
        "status": "validation_targets_met" if fine_metrics["acceptance_and_rejection"]["meets_targets"]
        else "validation_targets_not_met",
        "config": str(Path(config_path).resolve()),
        "manifest": str(manifest.resolve()),
        "resumed_from_checkpoint": str(resume_checkpoint),
        "resumed_checkpoint_sha256": checkpoint_hash_before,
        "head_only_report": str(head_report_path),
        "head_only_checkpoint_preserved": checkpoint_hash_before == checkpoint_hash_after,
        "head_only_stage_repeated": False,
        "classes": list(classes),
        "class_to_idx": config["class_to_idx"],
        "fine_tune": {
            "backbone_fraction_max": 0.20,
            "layer_trainability_check": fine_tune_model_checks,
            "trainable_backbone_layer_names": fine_layers,
            "batch_normalization_frozen": batch_norm_frozen,
            "learning_rate": 1e-5,
            "epochs_requested_max": max_epochs,
            "epochs_completed": fine_stage["epochs_completed"],
            "best_epoch": fine_stage["best_epoch"],
            "best_validation_loss": fine_stage["best_validation_loss"],
            "training_seconds": fine_stage["training_seconds"],
            "history_csv": fine_stage["history_csv"],
            "checkpoint": fine_stage["checkpoint"],
        },
        "comparison": _metric_comparison(head_metrics, fine_metrics),
        "head_only_validation": head_metrics,
        "fine_tuned_validation": fine_metrics,
        "fine_tuned_targets_met": fine_metrics["acceptance_and_rejection"]["meets_targets"],
        "selected_checkpoint": str(selected_checkpoint.resolve()),
        "selection_policy": "Select fine-tuned checkpoint only if its best validation loss is lower than the recorded head-only best validation loss; otherwise retain the head-only checkpoint.",
        "held_out_test": {
            "evaluated": False,
            "reason": "Fine-tune-only runner never reads or evaluates held-out test images.",
        },
        "tflite_export": {"performed": False},
        "tensorflow_version": tf.__version__,
        "seed": seed,
    }
    _atomic_json_new(output_dir / "fine_tune_report.json", report)
    return report


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training(
    *, config_path: Path = DEFAULT_CONFIG, output_dir: Path | None = None,
    allow_finetune: bool = False, base_model: tf.keras.Model | None = None,
    partitions: dict | None = None, max_head_epochs: int = 3,
    max_finetune_epochs: int = 2, seed: int = SEED,
) -> dict:
    """Train only on train rows, gate by validation, and never load test rows."""
    config = load_config(config_path)
    default_output = ROOT / config["output_directory"]
    output_dir = Path(output_dir) if output_dir is not None else default_output
    manifest, checkpoint, output_dir = _validate_config_paths(config, output_dir)
    classes = tuple(config["classes"])
    class_to_idx = config["class_to_idx"]
    partitions = partitions or load_fit_and_validation_rows(manifest, classes[:-1])
    train_rows = partitions["train_fit"]
    validation_fit_rows = partitions["validation_fit"]
    validation_eval_rows = (
        validation_fit_rows + partitions["validation_unsupported"]
    )
    supported_validation = partitions["validation_supported"]

    _set_seeds(seed)
    if base_model is None:
        base_model = tf.keras.models.load_model(checkpoint, compile=False)
    if tuple(base_model.output_shape) != (None, 13):
        raise ValueError("Initial checkpoint must retain the production 13-class output")

    # Preserve the original model for the supported-validation comparator.
    # It is not used for fitting and no test images are loaded by this runner.
    baseline = predict_rows(base_model, supported_validation)
    supported_indices = [class_to_idx[row["food_class"]] for row in supported_validation]
    baseline_accuracy = float(np.mean(baseline.argmax(axis=1) == supported_indices))

    model = build_14class_model(base_model, classes)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "class_mapping.json").write_text(
        json.dumps({"classes": list(classes), "class_to_idx": class_to_idx}, indent=2) + "\n",
        encoding="utf-8",
    )
    train_dataset = make_dataset(
        train_rows, class_to_idx, batch_size=32, training=True, seed=seed,
    )
    validation_dataset = make_dataset(
        validation_fit_rows, class_to_idx, batch_size=32, training=False, seed=seed,
    )

    head_checkpoint = output_dir / "phase9_3_14class_head_best.keras"
    head_history = output_dir / "phase9_3_14class_head_history.csv"
    head_stage = fit_stage(
        model, train_dataset, validation_dataset,
        checkpoint_path=head_checkpoint, history_path=head_history,
        epochs=min(max_head_epochs, 3), learning_rate=1e-3,
    )
    head_model = tf.keras.models.load_model(head_checkpoint, compile=False)
    head_probabilities = predict_rows(head_model, validation_eval_rows)
    head_metrics = validation_metrics(
        validation_eval_rows, head_probabilities, classes, baseline_accuracy,
        config["acceptance_targets"],
    )
    stages = {"frozen_head": {**head_stage, "validation": head_metrics}}
    selected_model_path = head_checkpoint
    selected_metrics = head_metrics

    if allow_finetune and not head_metrics["acceptance_and_rejection"]["meets_targets"]:
        fine_model = tf.keras.models.load_model(head_checkpoint, compile=False)
        fine_layers = unfreeze_last_backbone_fraction(fine_model, fraction=0.20)
        fine_checkpoint = output_dir / "phase9_3_14class_finetune_best.keras"
        fine_history = output_dir / "phase9_3_14class_finetune_history.csv"
        fine_stage = fit_stage(
            fine_model, train_dataset, validation_dataset,
            checkpoint_path=fine_checkpoint, history_path=fine_history,
            epochs=min(max_finetune_epochs, 2), learning_rate=1e-5,
        )
        best_fine_model = tf.keras.models.load_model(fine_checkpoint, compile=False)
        fine_probabilities = predict_rows(best_fine_model, validation_eval_rows)
        fine_metrics = validation_metrics(
            validation_eval_rows, fine_probabilities, classes, baseline_accuracy,
            config["acceptance_targets"],
        )
        stages["last_20_percent_finetune"] = {
            **fine_stage, "trainable_backbone_layer_names": fine_layers,
            "validation": fine_metrics,
        }
        if fine_stage["best_validation_loss"] < head_stage["best_validation_loss"]:
            selected_model_path = fine_checkpoint
            selected_metrics = fine_metrics

    report = {
        "experiment": config["experiment"],
        "status": "validation_candidate_passed" if selected_metrics["acceptance_and_rejection"]["meets_targets"]
        else "validation_targets_not_met",
        "config": str(Path(config_path).resolve()),
        "manifest": str(manifest.resolve()),
        "initial_checkpoint": str(checkpoint.resolve()),
        "classes": list(classes),
        "class_to_idx": class_to_idx,
        "decision_policy": selected_metrics["decision_policy"],
        "validation": selected_metrics,
        "stages": stages,
        "selected_checkpoint": str(selected_model_path.resolve()),
        "held_out_test": {
            "evaluated": False,
            "reason": "Training runner intentionally never loads or evaluates test images.",
        },
        "tflite_export": {
            "performed": False,
            "reason": "Production and release artifacts remain untouched; candidate export is a separate gated step.",
        },
        "test_evaluation_ready_only_if_validation_passes": bool(
            selected_metrics["acceptance_and_rejection"]["meets_targets"]
        ),
        "tensorflow_version": tf.__version__,
        "seed": seed,
        "input_contract": config["training_plan"]["input_contract"],
        "augmentation": config["training_plan"]["augmentation"],
        "maximum_epochs": min(max_head_epochs, 3)
        + (min(max_finetune_epochs, 2) if "last_20_percent_finetune" in stages else 0),
    }
    _atomic_json_new(output_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--head-report", type=Path)
    parser.add_argument("--run", action="store_true", help="Explicitly start the bounded training run")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Permit the separate last-20-percent stage only if head validation misses targets")
    parser.add_argument("--fine-tune-only", action="store_true",
                        help="Resume the recorded best 14-class head checkpoint; never repeat head-only training")
    args = parser.parse_args()
    if args.fine_tune_only and args.fine_tune:
        parser.error("--fine-tune-only and --fine-tune are separate modes")
    if args.fine_tune_only:
        try:
            config = load_config(args.config)
            resume, head_report, output, report = resolve_finetune_resume(
                config,
                resume_checkpoint=args.resume_checkpoint,
                head_report_path=args.head_report,
                output_dir=args.output_dir,
            )
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"Fine-tune-only preflight failed: {error}\n")
        if not args.run:
            checkpoint_hash_before = _sha256_file(resume)
            try:
                model, fine_layers, checks = prepare_finetune_model(
                    resume, tuple(config["classes"]),
                )
            except (OSError, ValueError, RuntimeError) as error:
                parser.exit(1, f"Fine-tune-only checkpoint inspection failed: {error}\n")
            del model
            if _sha256_file(resume) != checkpoint_hash_before:
                parser.exit(1, "Fine-tune-only preflight unexpectedly changed its source checkpoint\n")
            print(json.dumps({
                "training_started": False,
                "mode": "fine_tune_only_resume",
                "resume_checkpoint": str(resume),
                "head_only_report": str(head_report),
                "head_only_stage_repeated": False,
                "output_directory": str(output),
                "output_directory_available": not output.exists(),
                "head_only_status": report["status"],
                "fine_tune_layer_names": fine_layers,
                "fine_tune_trainability_checks": checks,
                "resume_checkpoint_unchanged": True,
                "run_command": (
                    "--fine-tune-only --run"
                    + (f" --resume-checkpoint {resume}" if args.resume_checkpoint else "")
                    + (f" --head-report {head_report}" if args.head_report else "")
                    + (f" --output-dir {output}" if args.output_dir else "")
                ),
            }, indent=2))
            return
        try:
            report = run_finetune_only(
                config_path=args.config,
                resume_checkpoint=resume,
                head_report_path=head_report,
                output_dir=output,
                max_epochs=2,
            )
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"Phase 9.3 fine-tune-only run failed: {error}\n")
        print(json.dumps({
            "status": report["status"],
            "fine_tune_checkpoint": report["fine_tune"]["checkpoint"],
            "selected_checkpoint": report["selected_checkpoint"],
            "fine_tune_report": str(output / "fine_tune_report.json"),
            "held_out_test_evaluated": report["held_out_test"]["evaluated"],
        }, indent=2))
        return
    if args.fine_tune and not args.run:
        parser.error("--fine-tune requires --run")
    if not args.run:
        config = load_config(args.config)
        print(json.dumps({
            "training_started": False,
            "message": "Training is opt-in; rerun with --run to start at most three frozen-backbone epochs.",
            "class_to_idx": config["class_to_idx"],
            "output_directory": str(args.output_dir or ROOT / config["output_directory"]),
        }, indent=2))
        return
    try:
        report = run_training(
            config_path=args.config,
            output_dir=args.output_dir,
            allow_finetune=args.fine_tune,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Phase 9.3 14-class training failed: {error}\n")
    print(json.dumps({
        "status": report["status"],
        "selected_checkpoint": report["selected_checkpoint"],
        "training_report": str((Path(report["selected_checkpoint"]).parent / "training_report.json")),
        "held_out_test_evaluated": report["held_out_test"]["evaluated"],
    }, indent=2))


if __name__ == "__main__":
    main()
