"""Train MobileNetV3 Small on the verified mixed Phase 5 13-class splits."""

import argparse
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
    from .mobilenet_v3 import IMAGE_SIZE, MODEL_NAME, create_model
    from .tf_phase5_data import create_tf_phase5_datasets
else:
    from dataset import PHASE5_CLASSES
    from mobilenet_v3 import IMAGE_SIZE, MODEL_NAME, create_model
    from tf_phase5_data import create_tf_phase5_datasets


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_CHECKPOINT = ARTIFACT_DIR / "mobilenet_v3_small_phase7_v2_initial_best.keras"
DEFAULT_HISTORY = ARTIFACT_DIR / "mobilenet_v3_small_phase7_v2_initial_training.json"
SELECTED_CHECKPOINT = ARTIFACT_DIR / "mobilenet_v3_small_phase7_v2_best.keras"
SELECTED_HISTORY = ARTIFACT_DIR / "mobilenet_v3_small_phase7_v2_training.json"


def _write_json_new(path: Path, payload: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing training record: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    *, local_root=None, food101_root=None, checkpoint=DEFAULT_CHECKPOINT,
    history_path=DEFAULT_HISTORY, epochs=5, batch_size=32, seed=42, learning_rate=1e-4,
    initial_checkpoint=None,
):
    checkpoint, history_path = Path(checkpoint).expanduser(), Path(history_path).expanduser()
    if checkpoint.exists() or history_path.exists():
        raise FileExistsError("Choose new output paths; this training command never overwrites artifacts")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    datasets, source_datasets, data_metadata = create_tf_phase5_datasets(
        local_root=local_root, food101_root=food101_root or Path(__file__).resolve().parents[1] / "data" / "raw",
        batch_size=batch_size, seed=seed,
    )
    if initial_checkpoint is None:
        model = create_model(pretrained=True)
    else:
        initial_checkpoint = Path(initial_checkpoint).expanduser().resolve()
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"Initial MobileNetV3 checkpoint is missing: {initial_checkpoint}")
        model = tf.keras.models.load_model(initial_checkpoint, compile=False)
        if tuple(model.output_shape) != (None, len(PHASE5_CLASSES)):
            raise ValueError("Initial MobileNetV3 checkpoint has an incompatible output head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint), monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=False, verbose=1,
    )
    history = model.fit(
        datasets["train"], validation_data=datasets["validation"], epochs=epochs,
        callbacks=[callback], verbose=2, shuffle=False,
    )
    history_values = {key: [float(value) for value in values] for key, values in history.history.items()}
    best_epoch = int(np.argmax(history_values["val_accuracy"])) + 1
    record = {
        "model_name": MODEL_NAME, "variant": "MobileNetV3Small", "phase": 7,
        "classes": list(PHASE5_CLASSES), "class_to_idx": {name: i for i, name in enumerate(PHASE5_CLASSES)},
        "image_size": IMAGE_SIZE, "input": "RGB float32 [0,255], NHWC; model contains [-1,1] rescaling",
        "pretrained": "Keras MobileNetV3Small ImageNet weights; all backbone layers fine-tuned",
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "head": "fresh dropout(0.2) + 13-way softmax dense layer",
        "optimizer": "Adam", "learning_rate": learning_rate,
        "loss": "SparseCategoricalCrossentropy", "batch_size": batch_size,
        "epochs_requested": epochs, "seed": seed, "selection_metric": "validation_accuracy",
        "best_epoch": best_epoch, "best_validation_accuracy": history_values["val_accuracy"][best_epoch - 1],
        "best_validation_loss": history_values["val_loss"][best_epoch - 1],
        "history": history_values, "data": data_metadata,
        "split_samples": {name: len(dataset) for name, dataset in source_datasets.items()},
        "checkpoint": str(checkpoint.resolve()), "tensorflow_version": tf.__version__,
    }
    _write_json_new(history_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--food101-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = train(
            local_root=args.local_root, food101_root=args.food101_root,
            checkpoint=args.checkpoint, history_path=args.history, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate,
            initial_checkpoint=args.initial_checkpoint,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"MobileNetV3 training failed: {error}\n")
    print(json.dumps({key: result[key] for key in (
        "checkpoint", "best_epoch", "best_validation_accuracy", "best_validation_loss", "split_samples",
    )}, indent=2))


if __name__ == "__main__":
    main()
