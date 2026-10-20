"""Train a separate 13-class model from Phase 3 features plus both food datasets."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .checkpoint import ARTIFACT_DIR, save_checkpoint
    from .dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from .model import MODEL_NAME, ModelConfig, create_model
    from .phase5_data import balanced_class_weights, class_counts, create_phase5_data_pipeline
    from .runtime import seed_training, select_device
    from .training import TrainingConfig, train_one_epoch, validate
else:
    from checkpoint import ARTIFACT_DIR, save_checkpoint
    from dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from model import MODEL_NAME, ModelConfig, create_model
    from phase5_data import balanced_class_weights, class_counts, create_phase5_data_pipeline
    from runtime import seed_training, select_device
    from training import TrainingConfig, train_one_epoch, validate

PHASE3_CHECKPOINT = ARTIFACT_DIR / "efficientnet_b0_best.pt"
PHASE5_CHECKPOINT = ARTIFACT_DIR / "efficientnet_b0_phase5_best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_from_phase3(model: nn.Module, checkpoint_path: Path) -> str:
    """Transfer only compatible EfficientNet feature weights, never the 10-way head."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_mapping = {name: index for index, name in enumerate(DEVELOPMENT_CLASSES)}
    if (checkpoint.get("model_name") != MODEL_NAME or checkpoint.get("classes") != list(DEVELOPMENT_CLASSES)
            or checkpoint.get("class_to_idx") != expected_mapping
            or checkpoint.get("model_config", {}).get("num_classes") != len(DEVELOPMENT_CLASSES)):
        raise ValueError("Initialization checkpoint is not the verified Phase 3 ten-class model")
    state = checkpoint["model_state_dict"]
    destination = model.state_dict()
    feature_state = {key: value for key, value in state.items() if key.startswith("features.")}
    if not feature_state or any(key not in destination or destination[key].shape != value.shape for key, value in feature_state.items()):
        raise ValueError("Phase 3 feature parameters are incompatible with Phase 5 model")
    if any(key.startswith("classifier.") for key in feature_state):
        raise RuntimeError("Refusing to load the Phase 3 classifier head")
    missing_features = {key for key in destination if key.startswith("features.")} - feature_state.keys()
    if missing_features:
        raise ValueError(f"Phase 3 checkpoint is missing feature weights: {sorted(missing_features)}")
    destination.update(feature_state)
    model.load_state_dict(destination, strict=True)
    return sha256(checkpoint_path)


def train_phase5(
    *, local_root: Path, food101_root: Path = DATA_DIR, checkpoint_path: Path = PHASE5_CHECKPOINT,
    epochs: int = 5, batch_size: int = 16, num_workers: int = 0, requested_device: str = "auto",
    seed: int = 42,
) -> dict:
    if Path(checkpoint_path).resolve() == PHASE3_CHECKPOINT.resolve():
        raise ValueError("Phase 5 checkpoint must not overwrite the historical Phase 3 checkpoint")
    seed_training(seed)
    device = select_device(requested_device)
    config = DataConfig(root=food101_root, classes=DEVELOPMENT_CLASSES, image_size=224,
                        batch_size=batch_size, num_workers=num_workers, random_seed=seed)
    datasets, loaders = create_phase5_data_pipeline(config, local_root=local_root)
    print("Training samples by class:", class_counts(datasets["train"]), flush=True)
    model_config = ModelConfig(num_classes=len(PHASE5_CLASSES), pretrained=False)
    model = create_model(model_config)
    source_digest = initialize_from_phase3(model, PHASE3_CHECKPOINT)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    weights = balanced_class_weights(datasets["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    training_config = TrainingConfig(epochs=epochs)
    checkpoint_data_config = DataConfig(
        root=food101_root, classes=PHASE5_CLASSES, image_size=224, batch_size=batch_size,
        num_workers=num_workers, random_seed=seed,
    )
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_metrics = validate(model, loaders["validation"], criterion, device)
        print(f"Epoch {epoch}/{epochs} | train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.2%} | "
              f"validation loss {validation_metrics.loss:.4f} acc {validation_metrics.accuracy:.2%}", flush=True)
        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, model_config=model_config,
                data_config=checkpoint_data_config,
                class_to_idx={name: index for index, name in enumerate(PHASE5_CLASSES)},
                training={
                    "epoch": epoch, "config": asdict(training_config),
                    "train_metrics": asdict(train_metrics),
                    "validation_metrics": asdict(validation_metrics),
                    "selection_metric": "validation_accuracy",
                    "initialization": "Phase 3 EfficientNet-B0 features only; new 13-class head initialized from torchvision EfficientNet-B0 architecture",
                    "phase3_checkpoint_sha256": source_digest,
                    "class_weighting": "inverse-frequency weights N/(13*n_class), normalized to mean 1",
                    "class_weights": weights.detach().cpu().tolist(),
                    "train_class_counts": class_counts(datasets["train"]),
                    "validation_class_counts": class_counts(datasets["validation"]),
                    "dataset_manifest": str((Path(local_root) / "manifest.json").resolve()),
                },
            )
    return {"checkpoint": str(checkpoint_path), "best_validation_accuracy": best_accuracy, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/local_foods/turkishfoods25_3class")
    parser.add_argument("--food101-root", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=PHASE5_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = train_phase5(local_root=args.local_root, food101_root=args.food101_root,
                              checkpoint_path=args.checkpoint, epochs=args.epochs,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              requested_device=args.device, seed=args.seed)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Phase 5 training failed: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
