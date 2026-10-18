"""Fine-tune EfficientNet-B0 using only Phase 2 train and validation loaders."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .checkpoint import DEFAULT_CHECKPOINT, save_checkpoint
    from .dataset import DataConfig, DEVELOPMENT_CLASSES, create_data_pipeline
    from .model import MODEL_NAME, ModelConfig, create_model
    from .runtime import seed_training, select_device
    from .training import TrainingConfig, train_one_epoch, validate
else:
    from checkpoint import DEFAULT_CHECKPOINT, save_checkpoint
    from dataset import DataConfig, DEVELOPMENT_CLASSES, create_data_pipeline
    from model import MODEL_NAME, ModelConfig, create_model
    from runtime import seed_training, select_device
    from training import TrainingConfig, train_one_epoch, validate


def run_training(
    data_config: DataConfig, training_config: TrainingConfig, *,
    requested_device: str = "auto", pretrained: bool = True,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> list[dict]:
    """Start a fresh run; keep the earliest epoch when validation accuracies tie.

    The shared pipeline constructs all split views, as in Phase 2. Only the train
    and validation loaders are consumed here; no test sample reaches the model.
    """
    if data_config.classes != DEVELOPMENT_CLASSES or data_config.image_size != 224:
        raise ValueError("Phase 3 training requires the existing ten classes and 224x224 inputs")
    device = select_device(requested_device)
    seed_training(data_config.random_seed)
    datasets, loaders = create_data_pipeline(data_config)
    train_data, validation_data = datasets["train"], datasets["validation"]
    if train_data.class_to_idx != validation_data.class_to_idx or train_data.classes != data_config.classes:
        raise ValueError("Training and validation class mappings disagree")
    model_config = ModelConfig(num_classes=len(train_data.classes), pretrained=pretrained)
    model = create_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay,
    )
    print(
        f"Model: {MODEL_NAME}; device: {device}; pretrained: {pretrained}; "
        f"train: {len(train_data):,}; validation: {len(validation_data):,}; "
        f"seed: {data_config.random_seed}", flush=True,
    )
    print(f"Best checkpoint for this run: {checkpoint_path} (replaced on improvement)", flush=True)
    best_accuracy = -1.0
    history = []
    for epoch in range(1, training_config.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_metrics = validate(model, loaders["validation"], criterion, device)
        history.append({
            "epoch": epoch, "train": asdict(train_metrics), "validation": asdict(validation_metrics),
        })
        improved = validation_metrics.accuracy > best_accuracy
        if improved:
            best_accuracy = validation_metrics.accuracy
            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, model_config=model_config,
                data_config=data_config, class_to_idx=train_data.class_to_idx,
                training={
                    "epoch": epoch, "config": asdict(training_config),
                    "train_metrics": asdict(train_metrics), "validation_metrics": asdict(validation_metrics),
                    "selection_metric": "validation_accuracy", "history": list(history),
                    "device": str(device), "optimizer": "AdamW", "loss": "CrossEntropyLoss",
                    "fine_tune": "all_layers",
                },
            )
        print(
            f"Epoch {epoch}/{training_config.epochs} | "
            f"train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.2%} | "
            f"val loss {validation_metrics.loss:.4f} acc {validation_metrics.accuracy:.2%}"
            + (" | saved best" if improved else ""), flush=True,
        )
    return history


def main() -> None:
    defaults = DataConfig()
    training_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=defaults.root)
    parser.add_argument("--epochs", type=int, default=training_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=training_defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=training_defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    arguments = parser.parse_args()
    try:
        data_config = DataConfig(
            root=arguments.root, batch_size=arguments.batch_size,
            num_workers=arguments.num_workers, random_seed=arguments.seed,
        )
        training_config = TrainingConfig(
            epochs=arguments.epochs, learning_rate=arguments.lr, weight_decay=arguments.weight_decay,
        )
        run_training(
            data_config, training_config, requested_device=arguments.device,
            pretrained=arguments.pretrained, checkpoint_path=arguments.checkpoint,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
