"""Fetch the pinned TurkishFoods-25 dataset and export only three mapped foods."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

DATASET_ID = "yunusserhat/TurkishFoods-25"
REVISION = "3e5026a17d14df81a25f0b6cdb16a58301fbb83c"
SOURCE_TO_CANONICAL = {"borek": "borek", "kebap": "kebap", "yaprak_sarma": "sarma"}
SPLIT_PRIORITY = {"test": 0, "eval": 1, "train": 2}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/local_foods/turkishfoods25_3class"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reuse_prepared_dataset(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FileExistsError(
            f"Prepared-data path exists without a readable manifest: {output}; choose a new --output"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
        raise ValueError(f"Prepared data manifest is malformed: {manifest_path}")
    if (manifest.get("dataset_id") != DATASET_ID or manifest.get("revision") != REVISION
            or manifest.get("source_to_canonical") != SOURCE_TO_CANONICAL):
        raise FileExistsError(f"Prepared data at {output} comes from another source revision or mapping")
    for item in manifest.get("records", []):
        path = (output / item.get("path", "")).resolve()
        if output not in path.parents or not path.is_file() or _sha256(path) != item.get("file_sha256"):
            raise ValueError(f"Prepared data is incomplete or changed: {path}")
    return manifest


def export_dataset(output: Path = DEFAULT_OUTPUT, *, dataset_dict=None) -> dict:
    """Keep pinned source splits, explicitly map names, and reject cross-split duplicates."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        return _reuse_prepared_dataset(output)
    if dataset_dict is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Install project requirements to use the TurkishFoods-25 exporter") from error
        dataset_dict = load_dataset(DATASET_ID, revision=REVISION)
    if set(dataset_dict) != set(SPLIT_PRIORITY):
        raise ValueError(f"Expected official train/eval/test splits, got {sorted(dataset_dict)}")
    feature = dataset_dict["train"].features.get("label")
    names = getattr(feature, "names", None)
    if not names or not set(SOURCE_TO_CANONICAL).issubset(names):
        raise ValueError("Dataset label feature does not contain the three expected source class names")
    expected_sizes = {"train": 9168, "eval": 1146, "test": 1147}
    if {name: len(dataset_dict[name]) for name in expected_sizes} != expected_sizes:
        raise ValueError("Pinned TurkishFoods-25 split sizes differ from the documented source revision")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".phase5-foods-", dir=output.parent))
    counts = {split: Counter() for split in SPLIT_PRIORITY}
    kept = {split: [] for split in SPLIT_PRIORITY}
    seen = {}
    duplicate_exclusions = []
    try:
        # Test wins a content collision so an image never migrates into training.
        for split in ("test", "eval", "train"):
            rows = dataset_dict[split]
            for row_index, row in enumerate(rows):
                source_label = row["label"]
                if isinstance(source_label, int):
                    source_label = names[source_label]
                if source_label not in SOURCE_TO_CANONICAL:
                    continue
                canonical = SOURCE_TO_CANONICAL[source_label]
                counts[split][canonical] += 1
                image = row["image"].convert("RGB")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                prior = seen.get(pixel_hash)
                if prior:
                    if prior["canonical_label"] != canonical:
                        raise ValueError(f"Identical image content has conflicting labels in {split}: {row_index}")
                    duplicate_exclusions.append({"split": split, "row_index": row_index, "kept": prior["source_split"]})
                    continue
                relative = Path(split) / canonical / f"{row_index:06d}.png"
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination, format="PNG", optimize=True)
                item = {
                    "path": relative.as_posix(), "source_split": split,
                    "source_label": source_label, "canonical_label": canonical,
                    "source_row": row_index, "pixel_sha256": pixel_hash,
                    "file_sha256": _sha256(destination),
                }
                kept[split].append(item)
                seen[pixel_hash] = item
        if any(not kept[split] for split in kept) or any(
            not any(item["canonical_label"] == label for item in kept[split])
            for split in kept for label in SOURCE_TO_CANONICAL.values()
        ):
            raise ValueError("One or more canonical food classes are missing from an official split")
        manifest = {
            "schema_version": 1, "dataset_id": DATASET_ID, "revision": REVISION,
            "dataset_url": f"https://huggingface.co/datasets/{DATASET_ID}",
            "license": "Apache-2.0", "source_splits": expected_sizes,
            "source_to_canonical": SOURCE_TO_CANONICAL,
            "source_class_names": names,
            "selected_source_classes": list(SOURCE_TO_CANONICAL),
            "source_selected_counts": {split: dict(counts[split]) for split in counts},
            "exported_counts": {split: dict(Counter(item["canonical_label"] for item in kept[split])) for split in kept},
            "duplicate_content_exclusions": duplicate_exclusions,
            "records": [item for split in ("train", "eval", "test") for item in kept[split]],
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = export_dataset(args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(json.dumps(manifest["exported_counts"], indent=2))
    print(f"Prepared dataset: {args.output}")


if __name__ == "__main__":
    main()
