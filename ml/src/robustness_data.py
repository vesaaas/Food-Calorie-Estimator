"""Leakage-aware image manifests and split utilities for robustness experiments.

The manifest is deliberately model-agnostic. It records the triage label and exact
supported food class separately, plus provenance fields used for grouping and
subgroup analysis. Images are never copied or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

from PIL import Image

if __package__:
    from .dataset import DATA_DIR, PHASE5_CLASSES, food101_samples, load_food101
    from .download_phase5_dataset import DEFAULT_OUTPUT
    from .phase5_data import validate_manifest
else:
    from dataset import DATA_DIR, PHASE5_CLASSES, food101_samples, load_food101
    from download_phase5_dataset import DEFAULT_OUTPUT
    from phase5_data import validate_manifest


TRIAGE_CLASSES = ("supported_food", "unsupported_food", "non_food")
FIELDNAMES = (
    "path", "split", "triage_label", "food_class", "source_dataset",
    "source_split", "source_id", "group_id", "scene_tags", "license",
    "duplicate_label_conflict", "source_url", "source_landing_url",
    "source_author", "source_author_url", "source_title", "source_original_md5",
    "source_original_size", "source_rotation", "source_thumbnail_url",
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "ml/data/processed/robustness_manifest.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dhash(path: Path) -> int:
    """64-bit difference hash used only to flag likely near duplicates."""
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8))
        pixels = list(resized.get_flattened_data())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _food_row(path: Path, *, label: str, source: str, source_split: str,
              source_id: str, split: str) -> dict[str, str]:
    triage = "supported_food" if label in PHASE5_CLASSES else "unsupported_food"
    return {
        "path": str(path.resolve()), "split": split, "triage_label": triage,
        "food_class": label,
        "source_dataset": source, "source_split": source_split,
        "source_id": source_id, "group_id": f"{source}:{source_id}",
        "scene_tags": "",
        "license": (
            "Food-101 research fair-use; verify source-image rights"
            if source == "Food-101" else "Apache-2.0 dataset provenance"
        ),
    }


def build_existing_manifest(*, food101_root: Path = DATA_DIR,
                            local_root: Path = DEFAULT_OUTPUT) -> list[dict[str, str]]:
    """Index the already-present Food-101 and TurkishFoods-25 files without downloading."""
    train, test = load_food101(food101_root)
    rows: list[dict[str, str]] = []
    for source, split_name, dataset, target_split in (
        ("Food-101", "train", train, "auto"),
        ("Food-101", "test", test, "test"),
    ):
        for path, class_index in food101_samples(dataset):
            label = dataset.classes[class_index]
            rows.append(_food_row(
                path, label=label, source=source, source_split=split_name,
                source_id=path.stem, split=target_split,
            ))

    local_manifest = validate_manifest(local_root)
    for item in local_manifest["records"]:
        source_split = item["source_split"]
        target_split = {"train": "auto", "eval": "validation", "test": "test"}[source_split]
        path = (Path(local_root) / item["path"]).resolve()
        canonical = item["canonical_label"]
        rows.append(_food_row(
            path, label=canonical, source="TurkishFoods-25", source_split=source_split,
            source_id=f"{source_split}:{item['source_row']}", split=target_split,
        ))
    return rows


def build_challenge_rows(root: Path) -> list[dict[str, str]]:
    """Index manually collected phone photos; `challenge` is never a train split."""
    root = Path(root).expanduser().resolve()
    rows = []
    for label in PHASE5_CLASSES:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        for path in sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            relative = path.relative_to(class_dir)
            session = relative.parts[0] if len(relative.parts) > 1 else "unspecified_session"
            variation = relative.parts[1] if len(relative.parts) > 2 else "unspecified_variation"
            rows.append({
                "path": str(path), "split": "challenge", "triage_label": "supported_food",
                "food_class": label, "source_dataset": "real_phone_challenge",
                "source_split": "challenge", "source_id": str(relative),
                "group_id": f"real_phone:{label}:{session}",
                "scene_tags": variation, "license": "owner_captured",
            })
    return rows


def assign_group_splits(rows: list[dict[str, str]], *, seed: int = 42,
                        validation_fraction: float = 0.1,
                        test_fraction: float = 0.0,
                        near_duplicate_distance: int = 5) -> list[dict[str, str]]:
    """Assign groups, never individual images, to train/validation/test.

    Existing assigned train/validation/test membership is frozen unless an
    exact/source/near-duplicate component crosses splits. In that case test wins,
    then validation, then train; conflicting rows are excluded. Records marked
    `auto` are split by triage label and exact supported class. Exact hashes and
    64-bit dHash candidates are joined before assignment. Source/session group IDs
    are joined regardless of label to prevent cross-label group leakage.
    """
    if not rows:
        raise ValueError("No image records were provided")
    if not 0 < validation_fraction < 0.5 or not 0 <= test_fraction < 0.5:
        raise ValueError("validation fraction must be in (0,0.5), test fraction in [0,0.5)")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation and test fractions leave too little training data")
    paths = [Path(row["path"]).expanduser().resolve() for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing image: {missing[0]}")
    union = _UnionFind(len(rows))
    by_group: dict[str, int] = {}
    by_hash: dict[str, int] = {}
    hashes, dhashes = [], []
    conflicting_exact_indices: set[int] = set()
    for index, (row, path) in enumerate(zip(rows, paths, strict=True)):
        group = row.get("group_id", "").strip()
        digest = row.get("sha256") or _sha256(path)
        perceptual = int(row["dhash"], 16) if row.get("dhash") else _dhash(path)
        hashes.append(digest)
        dhashes.append(perceptual)
        if group:
            if group in by_group:
                union.union(index, by_group[group])
            else:
                by_group[group] = index
        if digest in by_hash:
            prior = by_hash[digest]
            if (rows[index]["triage_label"], rows[index].get("food_class", "")) != (
                rows[prior]["triage_label"], rows[prior].get("food_class", "")
            ):
                conflicting_exact_indices.update((index, prior))
            union.union(index, by_hash[digest])
        else:
            by_hash[digest] = index

    # Eight 8-bit bands form a locality-sensitive index. Any pair within five
    # Hamming bits must share at least three bands; verify candidates exactly.
    bands: dict[tuple[int, int], list[int]] = {}
    for index, code in enumerate(dhashes):
        for band in range(8):
            key = (band, (code >> (band * 8)) & 0xFF)
            for prior in bands.get(key, ()):
                if _hamming(code, dhashes[prior]) <= near_duplicate_distance:
                    union.union(index, prior)
            bands.setdefault(key, []).append(index)

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(union.find(index), []).append(index)

    fixed_by_component: dict[int, set[str]] = {}
    fixed_winner: dict[int, str] = {}
    for root, members in components.items():
        fixed = {rows[i]["split"] for i in members if rows[i]["split"] in ("train", "validation", "test")}
        if fixed:
            # Keep the frozen supported-food test set intact and remove any
            # colliding train/validation records from this experiment.
            winner = "test" if "test" in fixed else "validation" if "validation" in fixed else "train"
            fixed_by_component[root] = {winner}
            fixed_winner[root] = winner

    component_split: dict[int, str] = {
        root: next(iter(splits)) for root, splits in fixed_by_component.items()
    }
    automatic = [root for root in components if root not in fixed_by_component]
    rng = random.Random(seed)
    rng.shuffle(automatic)
    labels = sorted({
        (rows[index]["triage_label"], rows[index].get("food_class", ""))
        for root in automatic for index in components[root]
        if rows[index]["split"] == "auto"
    })
    totals = {label: 0 for label in labels}
    group_counts: dict[int, dict[tuple[str, str], int]] = {}
    for root in automatic:
        counts: dict[tuple[str, str], int] = {}
        for index in components[root]:
            if rows[index]["split"] == "auto":
                label = (rows[index]["triage_label"], rows[index].get("food_class", ""))
                counts[label] = counts.get(label, 0) + 1
                totals[label] += 1
        group_counts[root] = counts
    targets = {
        "test": {label: totals[label] * test_fraction for label in labels},
        "validation": {label: totals[label] * validation_fraction for label in labels},
    }
    assigned = {split: {label: 0 for label in labels} for split in targets}
    for root in sorted(automatic, key=lambda item: -len(components[item])):
        counts = group_counts[root]
        candidates = ("test", "validation", "train")
        def cost(destination: str) -> float:
            value = 0.0
            for split in ("test", "validation"):
                for label in labels:
                    before = targets[split][label] - assigned[split][label]
                    after = before - (counts.get(label, 0) if split == destination else 0)
                    scale = max(1.0, targets[split][label])
                    value += (after / scale) ** 2 - (before / scale) ** 2
            return value
        destination = min(candidates, key=cost)
        component_split[root] = destination
        if destination in assigned:
            for label, count in counts.items():
                assigned[destination][label] += count
    output = []
    conflicting_roots = {union.find(index) for index in conflicting_exact_indices}
    for index, row in enumerate(rows):
        updated = dict(row)
        root = union.find(index)
        if root in fixed_winner and row["split"] != fixed_winner[root]:
            updated["split"] = "excluded"
        elif row["split"] == "auto":
            # If a development image duplicates a frozen evaluation image, exclude
            # the development copy instead of silently enlarging/moving the test set.
            updated["split"] = "excluded" if root in fixed_by_component else component_split[root]
        updated["sha256"] = hashes[index]
        updated["dhash"] = f"{dhashes[index]:016x}"
        updated["duplicate_label_conflict"] = str(root in conflicting_roots).lower()
        output.append(updated)
    return output


def write_csv(rows: Iterable[dict[str, str]], output: Path) -> None:
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FIELDNAMES) + ["sha256", "dhash"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _self_test() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="robustness_manifest_") as directory:
        root = Path(directory)
        rows = []
        for index, label in enumerate(("supported_food", "unsupported_food", "non_food")):
            image = Image.new("RGB", (40, 40), (index * 60, 50, 90))
            for y in range(40):
                for x in range(40):
                    image.putpixel((x, y), ((x * 17 + index * 31) % 256,
                                            (y * 23 + index * 47) % 256,
                                            (x * y + index * 61) % 256))
            path = root / f"{index}.png"
            image.save(path)
            rows.append({
                "path": str(path), "split": "auto", "triage_label": label,
                "food_class": "pizza" if index == 0 else "", "source_dataset": "synthetic",
                "source_split": "train", "source_id": str(index), "group_id": str(index),
                "scene_tags": "", "license": "test",
            })
        grouped = dict(rows[0])
        grouped["path"] = rows[0]["path"]
        grouped["source_id"] = "same-capture"
        grouped["group_id"] = "same-capture"
        rows[0]["group_id"] = "same-capture"
        rows.append(grouped)
        split = assign_group_splits(rows, seed=5, validation_fraction=0.1, test_fraction=0.15)
        assert len(split) == 4 and all(row["sha256"] and row["dhash"] for row in split)
        assert split[0]["split"] == split[3]["split"]
        assert all(row["split"] in {"train", "validation", "test"} for row in split)
    print("Robustness manifest self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index-existing", "index-challenge", "self-test"))
    parser.add_argument("--food101-root", type=Path, default=DATA_DIR)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--challenge-root", type=Path,
                        default=ROOT / "ml/data/real_phone_challenge")
    args = parser.parse_args()
    if args.command == "self-test":
        _self_test()
        return
    if args.command == "index-challenge":
        rows = build_challenge_rows(args.challenge_root)
        if not rows:
            parser.error(f"no challenge images found under {args.challenge_root}")
        write_csv(rows, args.output)
        print(json.dumps({"manifest": str(args.output), "challenge_images": len(rows)}, indent=2))
        return
    rows = build_existing_manifest(food101_root=args.food101_root, local_root=args.local_root)
    splits = assign_group_splits(rows)
    write_csv(splits, args.output)
    summary: dict[str, dict[str, int]] = {}
    for row in splits:
        summary.setdefault(row["split"], {}).setdefault(row["triage_label"], 0)
        summary[row["split"]][row["triage_label"]] += 1
    print(json.dumps({"manifest": str(args.output), "rows": len(splits), "counts": summary}, indent=2))


if __name__ == "__main__":
    main()
