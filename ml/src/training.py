"""Training and validation epochs, independent of device and dataset construction."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


BatchIterable = Iterable[tuple[torch.Tensor, torch.Tensor]]


def _run_epoch(
    model: nn.Module, batches: BatchIterable, criterion: nn.Module,
    device: torch.device, optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss; stopping before an optimizer/checkpoint update")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            samples += batch_size
    if samples == 0:
        raise ValueError("Cannot compute epoch metrics from an empty loader")
    return EpochMetrics(loss=total_loss / samples, accuracy=correct / samples, samples=samples)


def train_one_epoch(
    model: nn.Module, batches: BatchIterable, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> EpochMetrics:
    """Update parameters and report sample-weighted mean loss and accuracy."""
    return _run_epoch(model, batches, criterion, device, optimizer)


def validate(
    model: nn.Module, batches: BatchIterable, criterion: nn.Module, device: torch.device,
) -> EpochMetrics:
    """Evaluate validation data with eval mode and no gradients or optimizer updates."""
    return _run_epoch(model, batches, criterion, device)
