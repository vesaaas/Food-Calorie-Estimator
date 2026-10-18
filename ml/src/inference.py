"""Load a trained Food-101 classifier and predict the food in one image."""

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms

if __package__:
    from .checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
    from .preprocessing import build_transforms
    from .runtime import select_device
else:
    from checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
    from preprocessing import build_transforms
    from runtime import select_device


class FoodImagePredictor:
    """Reusable, read-only image predictor initialized from a validated checkpoint."""

    def __init__(
        self, model: nn.Module, *, class_to_idx: Mapping[str, int], image_size: int,
        device: torch.device,
    ) -> None:
        if not class_to_idx or set(class_to_idx.values()) != set(range(len(class_to_idx))):
            raise ValueError("Checkpoint class mapping must have unique contiguous indices")
        self.model = model.to(device)
        self.model.eval()
        self.class_to_idx = dict(class_to_idx)
        self.index_to_class = {index: name for name, index in class_to_idx.items()}
        self.image_size = image_size
        self.device = device
        self.transform: transforms.Compose = build_transforms(image_size)[1]

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path = DEFAULT_CHECKPOINT, *, requested_device: str = "auto",
    ) -> "FoodImagePredictor":
        device = select_device(requested_device)
        model, metadata = load_checkpoint(checkpoint_path, device=device)
        image_size = metadata["preprocessing"]["image_size"]
        if image_size != 224:
            raise ValueError(f"Food-101 inference expects the trained 224x224 input size, got {image_size}")
        if repr(build_transforms(image_size)[1]) != metadata["preprocessing"]["evaluation_transform"]:
            raise ValueError("Checkpoint evaluation transform does not match current preprocessing")
        return cls(
            model, class_to_idx=metadata["class_to_idx"],
            image_size=image_size, device=device,
        )

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Open a PIL-compatible image, convert it to RGB, and apply Phase 2 eval transforms."""
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.load()
                rgb_image = image.convert("RGB")
                return self.transform(rgb_image)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError(f"Could not read image '{image_path}': {error}") from error

    def predict(self, image_path: Path, *, top_k: int = 3) -> dict:
        """Return mapped class names and softmax confidence, with no gradient updates."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= len(self.class_to_idx):
            raise ValueError(f"top_k must be an integer from 1 to {len(self.class_to_idx)}")
        image = self.preprocess_image(image_path)
        if image.shape != (3, self.image_size, self.image_size):
            raise RuntimeError(f"Preprocessing returned unexpected image shape: {tuple(image.shape)}")
        batch = image.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
            if logits.shape != (1, len(self.class_to_idx)):
                raise RuntimeError(f"Expected [1, {len(self.class_to_idx)}] logits, got {list(logits.shape)}")
            probabilities = torch.softmax(logits[0], dim=0).cpu()
        if not torch.isfinite(probabilities).all().item() or not torch.isclose(
            probabilities.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6,
        ).item():
            raise RuntimeError("Model produced invalid class probabilities")
        top_indices = torch.topk(probabilities, k=top_k, largest=True, sorted=True).indices.tolist()
        top_predictions = [
            {
                "class_name": self.index_to_class[index], "class_index": index,
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ]
        best = top_predictions[0]
        return {
            "predicted_class": best["class_name"], "class_index": best["class_index"],
            "confidence": best["probability"], "top_predictions": top_predictions,
        }
