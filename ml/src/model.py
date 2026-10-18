"""EfficientNet-B0 for the existing Food-101 development labels."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

if __package__:
    from .dataset import DEVELOPMENT_CLASSES
else:
    from dataset import DEVELOPMENT_CLASSES


MODEL_NAME = "efficientnet_b0"
IMAGENET_WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = len(DEVELOPMENT_CLASSES)
    pretrained: bool = True
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.num_classes != len(DEVELOPMENT_CLASSES):
            raise ValueError("Phase 3 requires the existing ten development classes")
        if not isinstance(self.pretrained, bool):
            raise ValueError("pretrained must be a boolean")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def create_model(config: ModelConfig = ModelConfig()) -> nn.Module:
    """Fine-tune all layers; keep Phase 2 preprocessing and replace only the head.

    Pretrained initialization may download torchvision's pinned ImageNet weights.
    Loading our own checkpoint uses pretrained=False and never downloads weights.
    """
    weights = IMAGENET_WEIGHTS if config.pretrained else None
    if weights is not None:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        action = "Using cached" if cached.is_file() else "Downloading"
        print(f"{action} {MODEL_NAME} {weights.name} weights: {cached}", flush=True)
    try:
        model = efficientnet_b0(weights=weights, dropout=config.dropout)
    except (OSError, RuntimeError) as error:
        if weights is None:
            raise
        raise RuntimeError(
            f"Could not load {MODEL_NAME} ImageNet weights: {error}. "
            "Check network access and the torch hub cache (or TORCH_HOME). "
            "Use --no-pretrained only for an intentional random-weight run; "
            "no automatic fallback was performed."
        ) from error
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)
    return model
