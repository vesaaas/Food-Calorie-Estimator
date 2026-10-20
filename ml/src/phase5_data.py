"""Combine existing Food-101 development views with pinned local-food splits."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

if __package__:
    from .dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, Food101Subset, DataConfig, create_data_pipeline, seed_worker
    from .download_phase5_dataset import DEFAULT_OUTPUT, REVISION, SOURCE_TO_CANONICAL
    from .preprocessing import build_transforms
else:
    from dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, Food101Subset, DataConfig, create_data_pipeline, seed_worker
    from download_phase5_dataset import DEFAULT_OUTPUT, REVISION, SOURCE_TO_CANONICAL
    from preprocessing import build_transforms


def validate_manifest(root: Path, manifest_path: Path | None = None) -> dict:
    root = Path(root).resolve()
    manifest_path = manifest_path or root / "manifest.json"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("revision") != REVISION:
        raise ValueError("Unsupported local-food manifest or source revision")
    if manifest.get("source_to_canonical") != SOURCE_TO_CANONICAL:
        raise ValueError("Source-to-canonical mapping differs from the explicit Phase 5 mapping")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Local-food manifest has no records")
    content_by_split = {name: set() for name in ("train", "eval", "test")}
    paths = set()
    counts = {name: Counter() for name in content_by_split}
    for item in records:
        split, label = item.get("source_split"), item.get("canonical_label")
        if split not in content_by_split or label not in PHASE5_CLASSES[10:]:
            raise ValueError("Manifest contains an unexpected split or canonical label")
        if item.get("source_label") not in SOURCE_TO_CANONICAL or SOURCE_TO_CANONICAL[item["source_label"]] != label:
            raise ValueError("Manifest source label is not mapped to its canonical label explicitly")
        path = (root / item["path"]).resolve()
        if root not in path.parents or path in paths or not path.is_file():
            raise ValueError("Manifest image path is missing, duplicated, or escapes dataset root")
        paths.add(path)
        if _sha256(path) != item.get("file_sha256"):
            raise ValueError(f"Image file hash mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if pixel_hash != item.get("pixel_sha256") or pixel_hash in content_by_split[split]:
            raise ValueError("Manifest has invalid or duplicate image content")
        content_by_split[split].add(pixel_hash)
        counts[split][label] += 1
    split_names = tuple(content_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if content_by_split[left] & content_by_split[right]:
                raise ValueError(f"Identical image content leaks between local-food {left} and {right} splits")
    if any(not counts[name] for name in counts):
        raise ValueError("Every source split must contain selected regional classes")
    if any(counts[split] != Counter(manifest["exported_counts"][split]) for split in counts):
        raise ValueError("Manifest exported counts do not match its records")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalFoodDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, class_to_idx: dict[str, int]):
        self.root, self.records, self.transform, self.class_to_idx = Path(root), tuple(records), transform, class_to_idx
        self.targets = tuple(class_to_idx[item["canonical_label"]] for item in self.records)
        self.classes = PHASE5_CLASSES

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.root / self.records[index]["path"]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), self.targets[index]


class MixedFoodDataset(ConcatDataset):
    def __init__(self, food101: Food101Subset, regional: LocalFoodDataset, classes=PHASE5_CLASSES):
        super().__init__((food101, regional))
        self.classes = classes
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.targets = tuple(food101.targets) + tuple(regional.targets)


def create_phase5_data_pipeline(
    config: DataConfig, *, local_root: Path = DEFAULT_OUTPUT,
) -> tuple[dict, dict]:
    if tuple(config.classes) != DEVELOPMENT_CLASSES:
        raise ValueError("Food-101 data config must retain the established ten-class subset")
    manifest = validate_manifest(local_root)
    food_datasets, _ = create_data_pipeline(config)
    local_hashes = {item["pixel_sha256"]: item["source_split"] for item in manifest["records"]}
    for target_split, source_dataset in food_datasets.items():
        for image_path in source_dataset.image_paths:
            with Image.open(image_path) as image:
                pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            local_split = local_hashes.get(pixel_hash)
            if local_split is not None:
                expected = {"train": "train", "validation": "eval", "test": "test"}[target_split]
                if local_split != expected:
                    raise ValueError(
                        f"Identical content crosses Food-101 {target_split} and TurkishFoods-25 {local_split} splits"
                    )
    train_transform, evaluation_transform = build_transforms(config.image_size)
    mapping = {name: index for index, name in enumerate(PHASE5_CLASSES)}
    combined = {}
    for split, source_split in (("train", "train"), ("validation", "eval"), ("test", "test")):
        records = [item for item in manifest["records"] if item["source_split"] == source_split]
        local = LocalFoodDataset(local_root, records, train_transform if split == "train" else evaluation_transform, mapping)
        combined[split] = MixedFoodDataset(food_datasets[split], local)
    loaders = {
        split: DataLoader(
            ds, batch_size=config.batch_size, shuffle=split == "train", num_workers=config.num_workers,
            pin_memory=False, drop_last=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.random_seed + offset),
            multiprocessing_context="spawn" if config.num_workers else None,
        )
        for offset, (split, ds) in enumerate(combined.items())
    }
    return combined, loaders


def class_counts(dataset) -> dict[str, int]:
    return {name: dataset.targets.count(index) for index, name in enumerate(PHASE5_CLASSES)}


def balanced_class_weights(dataset) -> torch.Tensor:
    counts = class_counts(dataset)
    if any(count < 1 for count in counts.values()):
        raise ValueError(f"Training split lacks one or more classes: {counts}")
    weights = torch.tensor([len(dataset) / (len(counts) * counts[name]) for name in PHASE5_CLASSES], dtype=torch.float32)
    return weights / weights.mean()
