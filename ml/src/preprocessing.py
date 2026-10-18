"""CPU image transforms shared by the Food-101 data pipeline."""

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return mild training augmentation and deterministic evaluation transforms."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(
            round(image_size * 256 / 224),
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, evaluation_transform


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Return an RGB tensor in [0, 1] for visualization without changing the input."""
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    standard_deviation = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image.detach() * standard_deviation + mean).clamp(0, 1)
