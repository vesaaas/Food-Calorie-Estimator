"""Print Food-101 metadata and save a labeled, class-balanced image preview."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import random

import torch

if __package__:
    from .dataset import (
        PROCESSED_DIR, add_data_arguments, config_from_arguments, create_datasets,
        food101_samples, load_food101, seed_random_generators,
    )
    from .preprocessing import denormalize
else:
    from dataset import (
        PROCESSED_DIR, add_data_arguments, config_from_arguments, create_datasets,
        food101_samples, load_food101, seed_random_generators,
    )
    from preprocessing import denormalize


def save_preview(dataset, output: Path, view: str, samples: int, seed: int) -> None:
    """Save one sample per class first, showing raw or denormalized transformed RGB."""
    if samples < 1 or samples > len(dataset):
        raise ValueError(f"samples must be between 1 and {len(dataset)}")
    cache_directory = PROCESSED_DIR / ".matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    seed_random_generators(seed)
    generator = random.Random(seed)
    indices_by_class = [
        [index for index, target in enumerate(dataset.targets) if target == label]
        for label in range(len(dataset.classes))
    ]
    for indices in indices_by_class:
        generator.shuffle(indices)
    selected = []
    while len(selected) < samples:
        for indices in indices_by_class:
            if indices and len(selected) < samples:
                selected.append(indices.pop())
    columns = min(5, samples)
    rows = math.ceil(samples / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, selected):
        if view == "raw":
            image, _ = dataset.source[dataset.indices[index]]
            label = dataset.targets[index]
        else:
            tensor, label = dataset[index]
            image = denormalize(tensor).permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(f"{label}: {dataset.classes[label]}", fontsize=10)
    figure.suptitle(f"Food-101 development subset — {view}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print(f"Preview saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--view", choices=("raw", "train", "validation", "test"), default="validation")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "sample_grid.png")
    parser.add_argument("--no-plot", action="store_true", help="Print metadata only")
    arguments = parser.parse_args()
    config = config_from_arguments(arguments)
    torch.set_num_threads(1)
    official = load_food101(config.root)
    datasets = create_datasets(config, official)
    train_counts, test_counts = (
        Counter(source.classes[label] for _, label in food101_samples(source))
        for source in official
    )
    print(f"Total classes: {len(official[0].classes)}")
    print(f"Official training images: {len(official[0]):,}")
    print(f"Official test images: {len(official[1]):,}")
    print(f"{'Class':30} {'Train':>7} {'Test':>7}  Development")
    for name in official[0].classes:
        print(f"{name:30} {train_counts[name]:7} {test_counts[name]:7}  {'yes' if name in config.classes else ''}")
    print(f"Selected classes (label order): {list(config.classes)}")
    for split, dataset in datasets.items():
        print(f"Development {split}: {len(dataset):,}; per-class counts: {dict(sorted(Counter(dataset.targets).items()))}")
    if not arguments.no_plot:
        split = "train" if arguments.view == "raw" else arguments.view
        save_preview(datasets[split], arguments.output, arguments.view, arguments.samples, config.random_seed)


if __name__ == "__main__":
    main()
