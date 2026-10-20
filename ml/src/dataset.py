"""Validated Food-101 access, stratified index views, and CPU DataLoaders."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101

if __package__:
    from .preprocessing import build_transforms
else:
    from preprocessing import build_transforms


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = DATA_DIR.parent / "processed"
DEVELOPMENT_CLASSES = (
    "pizza", "hamburger", "sushi", "steak", "ice_cream", "donuts",
    "omelette", "french_fries", "chicken_wings", "spaghetti_bolognese",
)
REGIONAL_CLASSES = ("borek", "kebap", "sarma")
PHASE5_CLASSES = DEVELOPMENT_CLASSES + REGIONAL_CLASSES


@dataclass(frozen=True)
class DataConfig:
    root: Path = DATA_DIR
    classes: tuple[str, ...] = DEVELOPMENT_CLASSES
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    random_seed: int = 42
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        object.__setattr__(self, "classes", tuple(self.classes))
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be nonempty and unique")
        if self.image_size < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("image_size/batch_size must be positive; num_workers must be >= 0")
        if not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be in [0, 2**32)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")


def food101_samples(dataset: Food101) -> tuple[tuple[Path, int], ...]:
    """Adapt torchvision's private index fields in one place; never decode images here.

    Food101 has no public samples/targets properties. The pinned torchvision
    version exposes these fields; validation catches incompatible index changes.
    """
    if len(dataset._image_files) != len(dataset._labels):
        raise ValueError("Food-101 image and label index lengths differ")
    return tuple(zip(dataset._image_files, dataset._labels, strict=True))


def validate_food101(train: Food101, test: Food101) -> None:
    """Check metadata, every referenced file, class coverage, and official split isolation."""
    metadata_directory = Path(train.root) / "food-101" / "meta"
    declared_classes = (metadata_directory / "classes.txt").read_text().splitlines()
    if len(set(declared_classes)) != len(declared_classes):
        raise ValueError("Duplicate entries in Food-101 classes.txt")
    if sorted(declared_classes) != train.classes or train.classes != test.classes:
        raise ValueError("Food-101 class lists differ between metadata and official splits")
    if train.class_to_idx != test.class_to_idx:
        raise ValueError("Official splits have inconsistent label mappings")
    paths_by_split = []
    for split_name, source in (("train", train), ("test", test)):
        samples = food101_samples(source)
        counts = Counter(label for _, label in samples)
        if set(counts) != set(range(len(source.classes))):
            raise ValueError(f"Missing or invalid classes in official {split_name} split")
        image_directory = (Path(source.root) / "food-101" / "images").resolve()
        paths = set()
        for path, label in samples:
            resolved_path = path.resolve()
            expected_parent = image_directory / source.classes[label]
            if resolved_path.parent != expected_parent or resolved_path.suffix != ".jpg":
                raise ValueError(f"Unexpected image path or class: {path}")
            if not resolved_path.is_file() or resolved_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty Food-101 image: {path}")
            if resolved_path in paths:
                raise ValueError(f"Duplicate image in official {split_name} split: {path}")
            paths.add(resolved_path)
        paths_by_split.append(paths)
    if paths_by_split[0] & paths_by_split[1]:
        raise ValueError("Official train and test splits contain overlapping image paths")


def load_food101(root: Path = DATA_DIR, download: bool = False) -> tuple[Food101, Food101]:
    """Reuse valid local data; download only when explicitly requested and absent.

    Incomplete existing extractions fail validation without deleting or replacing
    data. The downloader offers explicit archive-based repair for that case.
    """
    root = Path(root).expanduser().resolve()
    if download:
        root.mkdir(parents=True, exist_ok=True)
    try:
        should_download = download and not (root / "food-101").exists()
        train = Food101(root=root, split="train", download=should_download)
        test = Food101(root=root, split="test", download=False)
        validate_food101(train, test)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"Food-101 at {root} could not be loaded/validated: {error}. "
            "Run download_dataset.py to download, or use its --repair option "
            "for an incomplete extraction."
        ) from error
    return train, test


class Food101Subset(Dataset):
    """An index-only view with contiguous labels and an independent transform."""

    def __init__(
        self, source: Food101, indices: list[int], classes: tuple[str, ...],
        transform: Callable,
    ) -> None:
        if source.transform is not None or source.target_transform is not None:
            raise ValueError("Subset source must have no transforms")
        self.source = source
        self.indices = tuple(indices)
        self.classes = classes
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        original_to_subset = {
            source.class_to_idx[name]: self.class_to_idx[name] for name in classes
        }
        samples = food101_samples(source)
        self.targets = tuple(original_to_subset[samples[index][1]] for index in indices)
        self.image_paths = tuple(samples[index][0] for index in indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.source[self.indices[index]]
        return self.transform(image.convert("RGB")), self.targets[index]


def create_datasets(
    config: DataConfig = DataConfig(),
    official: tuple[Food101, Food101] | None = None,
) -> dict[str, Food101Subset]:
    """Split each selected training class; keep every selected official test sample.

    An optional already-validated official pair avoids repeated filesystem scans.
    Classes retain the requested order, which defines the shared label mapping.
    """
    official_train, official_test = official if official is not None else load_food101(config.root)
    missing = set(config.classes) - set(official_train.classes)
    if missing:
        raise ValueError(f"Unknown Food-101 classes: {sorted(missing)}")
    train_by_class = {name: [] for name in config.classes}
    training_samples = food101_samples(official_train)
    for index, (_, label) in enumerate(training_samples):
        class_name = official_train.classes[label]
        if class_name in train_by_class:
            train_by_class[class_name].append(index)
    content_groups: dict[str, list[int]] = {}
    for indices in train_by_class.values():
        for index in indices:
            path = training_samples[index][0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            content_groups.setdefault(digest, []).append(index)
    groups_by_class = {name: [] for name in config.classes}
    for group in content_groups.values():
        for class_name in {official_train.classes[training_samples[index][1]] for index in group}:
            groups_by_class[class_name].append(group)
    train_indices, validation_indices = [], []
    assigned_to_validation: set[int] = set()
    for class_name, indices in train_by_class.items():
        validation_count = round(len(indices) * config.validation_fraction)
        if not 0 < validation_count < len(indices):
            raise ValueError(f"Split leaves an empty train/validation class: {class_name}")
        groups = sorted(
            groups_by_class[class_name],
            key=lambda group: training_samples[min(group)][0].as_posix(),
        )
        random.Random(f"{config.random_seed}:{class_name}").shuffle(groups)
        selected_count = 0
        for group in groups:
            if any(index in assigned_to_validation for index in group):
                continue
            group_size = sum(
                official_train.classes[training_samples[index][1]] == class_name
                for index in group
            )
            if selected_count + group_size <= validation_count:
                assigned_to_validation.update(group)
                validation_indices.extend(group)
                selected_count += group_size
            if selected_count == validation_count:
                break
        if selected_count != validation_count:
            raise ValueError(
                f"Cannot make an exact leakage-free validation split for {class_name}; "
                "duplicate-content groups conflict with the requested fraction"
            )
    train_indices = [
        index for class_name in config.classes for index in train_by_class[class_name]
        if index not in assigned_to_validation
    ]
    test_indices = [
        index for index, (_, label) in enumerate(food101_samples(official_test))
        if official_test.classes[label] in train_by_class
    ]
    train_transform, evaluation_transform = build_transforms(config.image_size)
    return {
        "train": Food101Subset(official_train, train_indices, config.classes, train_transform),
        "validation": Food101Subset(
            official_train, validation_indices, config.classes, evaluation_transform,
        ),
        "test": Food101Subset(official_test, test_indices, config.classes, evaluation_transform),
    }


def seed_random_generators(random_seed: int) -> None:
    """Seed main-process augmentation as well as Python/NumPy consumers."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def seed_worker(worker_id: int) -> None:
    """Picklable worker initializer for macOS spawn; torch seeds each worker itself."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(
    datasets: dict[str, Food101Subset], config: DataConfig = DataConfig(),
) -> dict[str, DataLoader]:
    """Build CPU loaders; only training shuffles, and no split drops its last batch.

    Call seed_random_generators before iteration for reproducible augmentation
    with num_workers=0. Worker seeds and shuffle order use separate generators.
    """
    return {
        split: DataLoader(
            dataset, batch_size=config.batch_size, shuffle=split == "train",
            num_workers=config.num_workers, pin_memory=False, drop_last=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.random_seed + offset),
            multiprocessing_context="spawn" if config.num_workers else None,
        )
        for offset, (split, dataset) in enumerate(datasets.items())
    }


def create_data_pipeline(
    config: DataConfig = DataConfig(),
) -> tuple[dict[str, Food101Subset], dict[str, DataLoader]]:
    """Create all three datasets and loaders; no download or device selection occurs."""
    seed_random_generators(config.random_seed)
    datasets = create_datasets(config)
    return datasets, create_dataloaders(datasets, config)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep the exploration and verification CLI defaults identical."""
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)


def config_from_arguments(arguments: argparse.Namespace) -> DataConfig:
    return DataConfig(
        root=arguments.root, image_size=arguments.image_size,
        batch_size=arguments.batch_size, num_workers=arguments.num_workers,
        random_seed=arguments.seed, validation_fraction=arguments.validation_fraction,
    )
