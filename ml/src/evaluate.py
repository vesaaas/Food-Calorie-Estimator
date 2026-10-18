"""Final held-out evaluation of a saved checkpoint. Run only after implementation review."""

import argparse
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

if __package__:
    from .checkpoint import ARTIFACT_DIR, DEFAULT_CHECKPOINT, load_checkpoint
    from .dataset import DataConfig, Food101Subset, create_data_pipeline, food101_samples
    from .runtime import select_device
else:
    from checkpoint import ARTIFACT_DIR, DEFAULT_CHECKPOINT, load_checkpoint
    from dataset import DataConfig, Food101Subset, create_data_pipeline, food101_samples
    from runtime import select_device


DEFAULT_REPORT = ARTIFACT_DIR / "test_evaluation.json"
OFFICIAL_TEST_SAMPLES_PER_CLASS = 250


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate_model(
    model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device, classes: tuple[str, ...],
) -> dict:
    """One read-only pass; confusion rows are true labels, columns are predictions.

    This metric routine also accepts synthetic batches for verification without
    accessing Food-101. The final-evaluation wrapper enforces official test coverage.
    """
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Evaluation classes must be nonempty and unique")
    num_classes = len(classes)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, samples = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in batches:
            if labels.ndim != 1 or labels.dtype != torch.int64 or labels.numel() == 0:
                raise ValueError("Expected a nonempty vector of int64 labels")
            if not bool(((labels >= 0) & (labels < num_classes)).all()):
                raise ValueError("Evaluation label is outside the checkpoint class mapping")
            logits = model(images.to(device))
            if logits.shape != (labels.size(0), num_classes):
                raise ValueError("Model output shape does not match the batch/class count")
            loss = criterion(logits, labels.to(device))
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite evaluation logits/loss")
            # Count on CPU so metrics do not depend on accelerator bincount support.
            predictions = logits.argmax(dim=1).cpu()
            cells = labels.cpu() * num_classes + predictions
            confusion += torch.bincount(cells, minlength=num_classes**2).reshape(num_classes, num_classes)
            total_loss += loss.item()
            samples += labels.size(0)
    if samples == 0:
        raise ValueError("Cannot evaluate an empty loader")
    per_class = {}
    for label, name in enumerate(classes):
        count = int(confusion[label].sum())
        correct = int(confusion[label, label])
        per_class[name] = {
            "label": label, "samples": count, "correct": correct,
            "accuracy": correct / count if count else None,
        }
    correct = int(confusion.diag().sum())
    return {
        "loss": total_loss / samples, "accuracy": correct / samples,
        "samples": samples, "correct": correct, "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
    }


def validate_test_loader(dataset: Food101Subset, loader: DataLoader, metadata: dict) -> None:
    """Check metadata/index coverage before decoding any official test images."""
    classes = tuple(metadata["classes"])
    expected_count = OFFICIAL_TEST_SAMPLES_PER_CLASS * len(classes)
    if dataset.classes != classes or dataset.class_to_idx != metadata["class_to_idx"]:
        raise ValueError("Test class order/mapping differs from the checkpoint")
    if getattr(dataset.source, "_split", None) != "test":
        raise ValueError("Final evaluation requires the official Food-101 test source")
    if repr(dataset.transform) != metadata["preprocessing"]["evaluation_transform"]:
        raise ValueError("Test preprocessing differs from the checkpoint's deterministic transforms")
    official = food101_samples(dataset.source)
    expected_indices = {
        index for index, (_, label) in enumerate(official)
        if dataset.source.classes[label] in classes
    }
    if len(dataset) != expected_count or len(set(dataset.indices)) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} unique official test samples")
    if set(dataset.indices) != expected_indices:
        raise ValueError("Test view does not contain every selected official test sample")
    expected_paths = tuple(official[index][0] for index in dataset.indices)
    expected_targets = tuple(
        metadata["class_to_idx"][dataset.source.classes[official[index][1]]]
        for index in dataset.indices
    )
    if dataset.image_paths != expected_paths or len(set(expected_paths)) != expected_count:
        raise ValueError("Test image paths are duplicated or inconsistent with official indices")
    if dataset.targets != expected_targets or Counter(dataset.targets) != Counter({
        label: OFFICIAL_TEST_SAMPLES_PER_CLASS for label in range(len(classes))
    }):
        raise ValueError("Expected 250 correctly mapped official test samples per class")
    if (
        loader.dataset is not dataset or type(loader.sampler) is not SequentialSampler
        or loader.sampler.data_source is not dataset or type(loader.batch_sampler) is not BatchSampler
        or loader.batch_sampler.sampler is not loader.sampler
        or loader.drop_last or loader.batch_sampler.drop_last
    ):
        raise ValueError("Test loader must visit every sample sequentially without replacement or dropping")


def validate_report_path(report_path: Path, checkpoint_path: Path) -> None:
    """Fail before inference if output could overwrite a checkpoint or prior report."""
    if report_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Evaluation report must not overwrite the checkpoint")
    if report_path.suffix.lower() != ".json":
        raise ValueError("Evaluation report must have a .json suffix")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"Report already exists: {report_path}; choose a new --report path")


def run_evaluation(
    checkpoint_path: Path = DEFAULT_CHECKPOINT, *, requested_device: str = "auto",
    batch_size: int = 16, num_workers: int = 0, root: Path | None = None,
    report_path: Path = DEFAULT_REPORT,
) -> dict:
    """Evaluate the official test loader once; never save or update model weights."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    report_path = Path(report_path).expanduser()
    validate_report_path(report_path, checkpoint_path)
    device = select_device(requested_device)
    checkpoint_digest = file_sha256(checkpoint_path)
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    saved_config = DataConfig(**metadata["data_config"])
    config = replace(
        saved_config, root=root if root is not None else saved_config.root,
        batch_size=batch_size, num_workers=num_workers,
    )
    datasets, loaders = create_data_pipeline(config)
    test_data, test_loader = datasets["test"], loaders["test"]
    validate_test_loader(test_data, test_loader, metadata)
    print(
        f"Evaluating checkpoint epoch {metadata['training']['epoch']} on {device}: "
        f"all {len(test_data):,} official test samples", flush=True,
    )
    metrics = evaluate_model(model, test_loader, device, tuple(metadata["classes"]))
    if metrics["samples"] != len(test_data) or any(
        item["samples"] != OFFICIAL_TEST_SAMPLES_PER_CLASS for item in metrics["per_class"].values()
    ):
        raise RuntimeError("Evaluation did not cover every official test sample exactly once")
    if file_sha256(checkpoint_path) != checkpoint_digest:
        raise RuntimeError("Checkpoint changed during evaluation; refusing to publish mismatched results")
    data_metadata = asdict(config)
    data_metadata.update(root=str(config.root), classes=list(config.classes))
    report = {
        "split": "official_test", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest, "checkpoint_epoch": metadata["training"]["epoch"],
        "checkpoint_validation_accuracy": metadata["training"]["validation_metrics"]["accuracy"],
        "model_name": metadata["model_name"], "model_config": metadata["model_config"],
        "classes": metadata["classes"], "class_to_idx": metadata["class_to_idx"],
        "data_config": data_metadata, "device": str(device), **metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing files, including symlink/hardlink aliases
    # of a checkpoint, even if the destination appears after the preflight check.
    with report_path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_report(report)
    print(f"Evaluation report: {report_path}", flush=True)
    return report


def print_report(report: dict) -> None:
    print(f"Test loss: {report['loss']:.6f} | accuracy: {report['accuracy']:.2%} | samples: {report['samples']:,}")
    print("Per-class accuracy (correct / true-class samples):")
    for name in report["classes"]:
        item = report["per_class"][name]
        print(f"  {name:24s} {item['accuracy']:7.2%}  ({item['correct']}/{item['samples']})")
    print("Confusion matrix: rows=true, columns=predicted; indices follow the class order above")
    print("     " + " ".join(f"{label:4d}" for label in range(len(report["classes"]))))
    for label, row in enumerate(report["confusion_matrix"]):
        print(f"{label:3d}: " + " ".join(f"{value:4d}" for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root", type=Path, help="Override the dataset root saved in the checkpoint")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    try:
        run_evaluation(
            arguments.checkpoint, requested_device=arguments.device,
            batch_size=arguments.batch_size, num_workers=arguments.num_workers,
            root=arguments.root, report_path=arguments.report,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
