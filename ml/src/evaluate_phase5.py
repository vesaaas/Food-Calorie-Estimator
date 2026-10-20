"""Evaluate a Phase 5 checkpoint on the independent mixed test splits once."""

import argparse
import json
from pathlib import Path

if __package__:
    from .checkpoint import ARTIFACT_DIR, load_checkpoint
    from .dataset import DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from .evaluate import evaluate_model, file_sha256
    from .phase5_data import class_counts, create_phase5_data_pipeline
    from .runtime import select_device
else:
    from checkpoint import ARTIFACT_DIR, load_checkpoint
    from dataset import DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from evaluate import evaluate_model, file_sha256
    from phase5_data import class_counts, create_phase5_data_pipeline
    from runtime import select_device

DEFAULT_CHECKPOINT = ARTIFACT_DIR / "efficientnet_b0_phase5_best.pt"
DEFAULT_REPORT = ARTIFACT_DIR / "phase5_mixed_test.json"


def run(checkpoint=DEFAULT_CHECKPOINT, report=DEFAULT_REPORT, *, device_name="auto", batch_size=16, local_root=None, food101_root=None):
    checkpoint, report = Path(checkpoint).resolve(), Path(report).expanduser()
    if report.exists() or report.is_symlink() or report.resolve() == checkpoint:
        raise FileExistsError(f"Refusing to overwrite checkpoint or existing report: {report}")
    device = select_device(device_name)
    digest = file_sha256(checkpoint)
    model, metadata = load_checkpoint(checkpoint, device=device, expected_classes=PHASE5_CLASSES)
    data = metadata["data_config"]
    config = DataConfig(root=food101_root or data["root"], classes=DEVELOPMENT_CLASSES,
                        image_size=data["image_size"], batch_size=batch_size,
                        num_workers=data["num_workers"], random_seed=data["random_seed"])
    datasets, loaders = create_phase5_data_pipeline(config, local_root=local_root or Path(metadata["training"]["dataset_manifest"]).parent)
    metrics = evaluate_model(model, loaders["test"], device, PHASE5_CLASSES)
    if digest != file_sha256(checkpoint):
        raise RuntimeError("Checkpoint changed during Phase 5 evaluation")
    result = {"phase": 5, "split": "Food-101 official test + TurkishFoods-25 source test",
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
              "checkpoint_epoch": metadata["training"]["epoch"], "device": str(device),
              "classes": list(PHASE5_CLASSES), "class_to_idx": metadata["class_to_idx"],
              "dataset_counts": {name: class_counts(dataset) for name, dataset in datasets.items()}, **metrics}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Phase 5 test loss {metrics['loss']:.5f}; accuracy {metrics['accuracy']:.2%}; samples {metrics['samples']}")
    for name in PHASE5_CLASSES:
        item = metrics["per_class"][name]
        print(f"{name:24s} {item['accuracy'] if item['accuracy'] is not None else float('nan'):.2%} ({item['correct']}/{item['samples']})")
    print(f"Report: {report}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--food101-root", type=Path)
    args = parser.parse_args()
    try:
        run(args.checkpoint, args.report, device_name=args.device, batch_size=args.batch_size,
            local_root=args.local_root, food101_root=args.food101_root)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Phase 5 evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
