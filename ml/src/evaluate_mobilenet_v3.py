"""Evaluate the selected MobileNetV3 model on the untouched Phase 5 mixed test split."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
    from .train_mobilenet_v3 import SELECTED_CHECKPOINT
    from .tf_phase5_data import create_tf_phase5_datasets
else:
    from dataset import PHASE5_CLASSES
    from train_mobilenet_v3 import SELECTED_CHECKPOINT
    from tf_phase5_data import create_tf_phase5_datasets


DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "artifacts" / "mobilenet_v3_small_phase7_v2_test.json"


def predict_dataset(model, dataset, expected_count: int):
    probabilities, labels = [], []
    for images, batch_labels in dataset:
        output = model(images, training=False).numpy()
        probabilities.append(output)
        labels.extend(batch_labels.numpy().tolist())
    probabilities = np.concatenate(probabilities, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != expected_count or probabilities.shape != (expected_count, len(PHASE5_CLASSES)):
        raise ValueError("Evaluation dataset yielded an unexpected number of images or classes")
    return labels, probabilities


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(PHASE5_CLASSES):
        raise ValueError("Output probabilities do not match the 13-class contract")
    predictions = probabilities.argmax(axis=1)
    epsilon = np.finfo(np.float32).tiny
    loss = float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], epsilon)).mean())
    per_class = {}
    for class_index, label in enumerate(PHASE5_CLASSES):
        selected = labels == class_index
        correct = int(np.sum(predictions[selected] == class_index))
        count = int(np.sum(selected))
        per_class[label] = {"correct": correct, "samples": count, "accuracy": correct / count if count else None}
    return {
        "samples": int(len(labels)), "loss": loss,
        "accuracy": float(np.mean(predictions == labels)), "per_class": per_class,
    }


def run(
    *, checkpoint=SELECTED_CHECKPOINT, report=DEFAULT_REPORT,
    local_root=None, food101_root=None, batch_size=32,
):
    checkpoint, report = Path(checkpoint).expanduser().resolve(), Path(report).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained MobileNetV3 checkpoint is missing: {checkpoint}")
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {report}")
    datasets, source_datasets, metadata = create_tf_phase5_datasets(
        local_root=local_root, food101_root=food101_root or Path(__file__).resolve().parents[1] / "data" / "raw",
        batch_size=batch_size, seed=42,
    )
    model = tf.keras.models.load_model(checkpoint, compile=False)
    labels, probabilities = predict_dataset(model, datasets["test"], len(source_datasets["test"]))
    result = {
        "phase": "MobileNetV3 TFLite deployment model", "split": "independent Phase 5 mixed test split",
        "checkpoint": str(checkpoint), "classes": list(PHASE5_CLASSES),
        "class_to_idx": {name: index for index, name in enumerate(PHASE5_CLASSES)},
        "dataset_counts": metadata["class_counts"]["test"], **metrics(labels, probabilities),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=report.parent, prefix=f".{report.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(report)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"MobileNetV3 test: {result['accuracy']:.2%}, loss {result['loss']:.5f}, n={result['samples']}")
    for label in PHASE5_CLASSES:
        row = result["per_class"][label]
        print(f"{label:24s} {row['accuracy']:.2%} ({row['correct']}/{row['samples']})")
    print(f"Report: {report}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--food101-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        run(checkpoint=args.checkpoint, report=args.report, local_root=args.local_root,
            food101_root=args.food101_root, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"MobileNetV3 evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
