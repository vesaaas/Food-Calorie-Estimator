"""Verify real Food-101 splits, transforms, and loaders without a model."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
import torchvision

if __package__:
    from .dataset import (
        DEVELOPMENT_CLASSES, PROCESSED_DIR, add_data_arguments, config_from_arguments,
        create_data_pipeline, create_dataloaders, create_datasets, food101_samples,
        seed_random_generators,
    )
    from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
else:
    from dataset import (
        DEVELOPMENT_CLASSES, PROCESSED_DIR, add_data_arguments, config_from_arguments,
        create_data_pipeline, create_dataloaders, create_datasets, food101_samples,
        seed_random_generators,
    )
    from preprocessing import IMAGENET_MEAN, IMAGENET_STD


def require(condition: bool, message: str) -> None:
    """Keep checks active even when Python runs with optimization enabled."""
    if not condition:
        raise AssertionError(message)


def verify_pipeline(config, all_batches: bool = False) -> dict:
    datasets, loaders = create_data_pipeline(config)
    official = datasets["train"].source, datasets["test"].source
    expected_mapping = {name: index for index, name in enumerate(config.classes)}
    require(config.classes == DEVELOPMENT_CLASSES, "Verification expects the ten development classes")
    paths = {split: set(dataset.image_paths) for split, dataset in datasets.items()}
    require(not paths["train"] & paths["validation"], "Train/validation overlap")
    require(not (paths["train"] | paths["validation"]) & paths["test"], "Test leakage")
    for split, dataset in datasets.items():
        require(len(paths[split]) == len(dataset), f"Duplicate samples in {split}")
        require(dataset.class_to_idx == expected_mapping, f"Incorrect mapping in {split}")
        require(set(dataset.targets) == set(expected_mapping.values()), f"Missing classes in {split}")
        for path, label in zip(dataset.image_paths, dataset.targets, strict=True):
            require(path.parent.name == config.classes[label], f"Wrong label for {path}")
    selected_training = {
        path for path, label in food101_samples(official[0])
        if official[0].classes[label] in config.classes
    }
    selected_test = {
        path for path, label in food101_samples(official[1])
        if official[1].classes[label] in config.classes
    }
    require(paths["train"] | paths["validation"] == selected_training, "Training partition is incomplete")
    require(paths["test"] == selected_test, "Official test selection changed")
    content_splits = {}
    for split, split_paths in paths.items():
        for path in split_paths:
            with path.open("rb") as image_file:
                digest = hashlib.file_digest(image_file, "sha256").hexdigest()
            previous_split = content_splits.setdefault(digest, split)
            require(previous_split == split, f"Identical image bytes across {previous_split}/{split}: {path}")
    counts = {split: Counter(dataset.targets) for split, dataset in datasets.items()}
    for label, name in enumerate(config.classes):
        total = counts["train"][label] + counts["validation"][label]
        require(
            counts["validation"][label] == round(total * config.validation_fraction),
            f"Incorrect stratification for {name}",
        )
    repeated = create_datasets(config, official)
    changed = create_datasets(replace(config, random_seed=(config.random_seed + 1) % 2**32), official)
    for split in datasets:
        require(datasets[split].indices == repeated[split].indices, f"Unstable {split} split")
    require(datasets["train"].indices != changed["train"].indices, "Seed does not change training split")
    require(datasets["test"].indices == changed["test"].indices, "Seed changes official test samples")
    for split in ("validation", "test"):
        for label in expected_mapping.values():
            index = datasets[split].targets.index(label)
            first, _ = datasets[split][index]
            second, _ = datasets[split][index]
            require(torch.equal(first, second), f"Random evaluation transform in {split}")
    print("PASS: class mapping, stratification, full partition, path/content split isolation, and deterministic evaluation")
    print("PASS: repeated seed reproduces splits; changing seed preserves official test selection")
    seed_random_generators(config.random_seed)
    batch_shapes = {}
    checked_samples = {}
    first_train_batch = None
    minimum = (-torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD)).view(1, 3, 1, 1)
    maximum = ((1 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)).view(1, 3, 1, 1)
    for split, loader in loaders.items():
        observed_labels = []
        for batch_number, (images, labels) in enumerate(loader):
            actual_batch_size = len(labels)
            require(images.shape == (actual_batch_size, 3, config.image_size, config.image_size), f"Bad {split} image shape")
            require(labels.shape == (actual_batch_size,), f"Bad {split} label shape")
            require(images.dtype == torch.float32 and labels.dtype == torch.int64, f"Bad {split} dtype")
            require(bool(torch.isfinite(images).all()), f"Nonfinite {split} image tensor")
            require(bool(((images >= minimum - 1e-6) & (images <= maximum + 1e-6)).all()), f"Unexpected normalized range in {split}")
            require(bool(((labels >= 0) & (labels < len(config.classes))).all()), f"Invalid {split} labels")
            require(images.device.type == labels.device.type == "cpu", "Loader selected a device")
            observed_labels.extend(labels.tolist())
            if batch_number == 0:
                batch_shapes[split] = {"images": list(images.shape), "labels": list(labels.shape)}
                require(actual_batch_size == min(config.batch_size, len(datasets[split])), "Wrong batch size")
                print(f"{split}: {len(datasets[split]):,} samples; images shape: {list(images.shape)}; labels shape: {list(labels.shape)}")
                if split == "train":
                    first_train_batch = images.clone(), labels.clone()
            if not all_batches:
                break
        checked_samples[split] = len(observed_labels)
        if all_batches:
            require(Counter(observed_labels) == counts[split], f"Loader drops/repeats samples in {split}")
        if split != "train":
            require(observed_labels == list(datasets[split].targets[:len(observed_labels)]), f"Evaluation loader shuffled {split}")
    seed_random_generators(config.random_seed)
    repeat_loader = create_dataloaders(datasets, config)["train"]
    repeat_images, repeat_labels = next(iter(repeat_loader))
    require(torch.equal(first_train_batch[0], repeat_images), "Augmentation is not seed-reproducible")
    require(torch.equal(first_train_batch[1], repeat_labels), "Shuffle is not seed-reproducible")
    print("PASS: CPU DataLoaders, tensor shapes/dtypes/ranges, and seeded first-batch reproduction")
    if all_batches:
        print(f"PASS: decoded and transformed all {sum(checked_samples.values()):,} selected images")
    return {
        "status": "passed", "torch": torch.__version__, "torchvision": torchvision.__version__,
        "root": str(config.root), "class_to_idx": expected_mapping,
        "config": {
            "image_size": config.image_size, "batch_size": config.batch_size,
            "num_workers": config.num_workers, "random_seed": config.random_seed,
            "validation_fraction": config.validation_fraction,
        },
        "sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "per_class_counts": {
            split: {name: counter[label] for label, name in enumerate(config.classes)}
            for split, counter in counts.items()
        },
        "split_sha256": {
            split: hashlib.sha256("\n".join(
                sorted(path.relative_to(config.root).as_posix() for path in split_paths)
            ).encode()).hexdigest()
            for split, split_paths in paths.items()
        },
        "batch_shapes": batch_shapes, "checked_samples": checked_samples,
        "all_batches": all_batches, "cross_split_identical_files": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--all-batches", action="store_true", help="Decode every selected image through its loader")
    parser.add_argument("--report", type=Path, default=PROCESSED_DIR / "verification.json")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = verify_pipeline(config_from_arguments(arguments), arguments.all_batches)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification report: {arguments.report}")


if __name__ == "__main__":
    main()
