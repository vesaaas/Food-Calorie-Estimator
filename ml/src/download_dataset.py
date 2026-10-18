"""Download once, validate local Food-101, or explicitly repair its extraction."""

import argparse
from pathlib import Path

from torchvision.datasets import Food101
from torchvision.datasets.utils import download_and_extract_archive

if __package__:
    from .dataset import DATA_DIR, load_food101
else:
    from dataset import DATA_DIR, load_food101


def download_food101(root: Path = DATA_DIR, repair: bool = False) -> None:
    """Reuse existing images; --repair re-extracts a checksum-verified archive."""
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {root}", flush=True)
    if repair:
        print("Repairing extraction from the official checksum-verified archive...", flush=True)
        download_and_extract_archive(Food101._URL, download_root=root, md5=Food101._MD5)
    train_dataset, test_dataset = load_food101(root, download=True)
    print("\nFood-101 is ready; metadata and all referenced image paths validated.")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--repair", action="store_true", help="Re-extract a damaged/incomplete dataset")
    arguments = parser.parse_args()
    download_food101(arguments.root, arguments.repair)
