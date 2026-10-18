"""Portable, metadata-validated best-model checkpoints (no weight downloads on load)."""

from dataclasses import asdict, replace
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
import torchvision

if __package__:
    from .dataset import DataConfig, DEVELOPMENT_CLASSES
    from .model import IMAGENET_WEIGHTS, MODEL_NAME, ModelConfig, create_model
    from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, build_transforms
else:
    from dataset import DataConfig, DEVELOPMENT_CLASSES
    from model import IMAGENET_WEIGHTS, MODEL_NAME, ModelConfig, create_model
    from preprocessing import IMAGENET_MEAN, IMAGENET_STD, build_transforms


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_CHECKPOINT = ARTIFACT_DIR / "efficientnet_b0_best.pt"
CHECKPOINT_VERSION = 1


def preprocessing_metadata(image_size: int) -> dict:
    """Describe the actual Phase 2 transforms without defining another pipeline."""
    training, evaluation = build_transforms(image_size)
    return {
        "image_size": image_size, "color_mode": "RGB",
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
        "training_transform": repr(training), "evaluation_transform": repr(evaluation),
    }


def _validate_metadata(checkpoint: dict, expected_classes: tuple[str, ...]) -> ModelConfig:
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format/version")
    required = {
        "model_name", "model_config", "pretrained_weights", "classes", "class_to_idx",
        "preprocessing", "data_config", "training", "model_state_dict", "optimizer_state_dict",
    }
    if not required <= checkpoint.keys():
        raise ValueError(f"Checkpoint is missing fields: {sorted(required - checkpoint.keys())}")
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError(f"Expected checkpoint model {MODEL_NAME}")
    try:
        model_config = ModelConfig(**checkpoint["model_config"])
        data_config = DataConfig(**checkpoint["data_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint configuration: {error}") from error
    classes = list(expected_classes)
    if checkpoint["classes"] != classes or list(data_config.classes) != classes:
        raise ValueError("Checkpoint class order differs from the expected Phase 2 class order")
    if checkpoint["class_to_idx"] != {name: index for index, name in enumerate(classes)}:
        raise ValueError("Checkpoint class_to_idx is inconsistent with its ordered classes")
    if model_config.num_classes != len(classes):
        raise ValueError("Checkpoint classifier size differs from its class count")
    weights = IMAGENET_WEIGHTS.name if model_config.pretrained else None
    if checkpoint["pretrained_weights"] != weights:
        raise ValueError("Checkpoint pretrained-weight metadata is inconsistent")
    if checkpoint["preprocessing"] != preprocessing_metadata(data_config.image_size):
        raise ValueError("Checkpoint preprocessing differs from the current Phase 2 transforms")
    training = checkpoint["training"]
    if not isinstance(training, dict) or not {
        "epoch", "config", "train_metrics", "validation_metrics", "selection_metric",
    } <= training.keys():
        raise ValueError("Checkpoint is missing training metadata")
    if not isinstance(training["epoch"], int) or training["epoch"] < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    if training["selection_metric"] != "validation_accuracy":
        raise ValueError("Checkpoint must be selected by validation accuracy")
    for name in ("train_metrics", "validation_metrics"):
        metrics = training[name]
        if not isinstance(metrics, dict) or not {"loss", "accuracy", "samples"} <= metrics.keys():
            raise ValueError(f"Invalid checkpoint {name}")
        if not math.isfinite(metrics["loss"]) or not 0 <= metrics["accuracy"] <= 1 or metrics["samples"] < 1:
            raise ValueError(f"Invalid checkpoint {name} values")
    return model_config


def save_checkpoint(
    path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
    model_config: ModelConfig, data_config: DataConfig,
    class_to_idx: dict[str, int], training: dict,
) -> None:
    """Atomically replace a checkpoint after a validation-accuracy improvement."""
    data_metadata = asdict(data_config)
    data_metadata.update(root=str(data_config.root), classes=list(data_config.classes))
    checkpoint = {
        "format_version": CHECKPOINT_VERSION, "model_name": MODEL_NAME,
        "model_config": asdict(model_config),
        "pretrained_weights": IMAGENET_WEIGHTS.name if model_config.pretrained else None,
        "classes": list(data_config.classes), "class_to_idx": dict(class_to_idx),
        "preprocessing": preprocessing_metadata(data_config.image_size),
        "data_config": data_metadata, "training": training,
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__)},
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    _validate_metadata(checkpoint, DEVELOPMENT_CLASSES)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, *, device: torch.device | str = "cpu",
    expected_classes: tuple[str, ...] = DEVELOPMENT_CLASSES,
) -> tuple[nn.Module, dict]:
    """Restore in eval mode with strict state loading and safe torch deserialization.

    CPU mapping makes saved accelerator tensors portable. Original pretrained
    provenance is retained, but reconstruction never fetches ImageNet weights.
    The optimizer state is returned for inspection; there is no exact-resume CLI.
    """
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    config = _validate_metadata(checkpoint, expected_classes)
    model = create_model(replace(config, pretrained=False))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint
