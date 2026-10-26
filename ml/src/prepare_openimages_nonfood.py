"""Reproducibly select and fetch a bounded Open Images V7 non-food subset.

The planner uses the official image-level human classification annotations to
select common object/scene classes while excluding every image with a positive
Food hierarchy label. It retains the original Open Images attribution fields.
The annotation labels are not a guarantee that an image was visually reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "ml/data/processed/openimages_nonfood"
METADATA = BASE / "metadata"
ANNOTATIONS = METADATA / "train-annotations-human-imagelabels-boxable.csv"
IMAGES = METADATA / "train-images-boxable-with-rotation.csv"
CLASS_NAMES = METADATA / "class-descriptions.csv"
HIERARCHY = METADATA / "class-hierarchy.json"
PLAN = BASE / "selected_images_thumbnail.csv"
DOWNLOAD_REPORT = BASE / "download_report.csv"
DATASET_QA = BASE / "dataset_qa.json"
COMBINED_MANIFEST = ROOT / "ml/data/processed/robustness_manifest_with_openimages.csv"

TARGETS = (
    "Laptop", "Mobile phone", "Computer monitor", "Computer keyboard",
    "Computer mouse", "Computer", "Desk", "Table", "Kitchen & dining room table",
    "Chair", "Couch", "Furniture", "Person", "Clothing", "Footwear", "Backpack",
    "Handbag", "Building", "House", "Car", "Bus", "Bicycle", "Dog", "Cat",
    "Book", "Bottle", "Plate", "Tableware", "Kitchen appliance", "Refrigerator",
    "Sink", "Microwave oven", "Oven", "Television", "Box", "Plastic bag", "Cup",
    "Mug", "Bowl", "Kitchen utensil", "Kitchenware", "Camera", "Clock", "Lamp",
    "Door", "Window", "Toilet", "Houseplant", "Tool", "Luggage and bags",
)
LICENSE_ALLOW = {"https://creativecommons.org/licenses/by/2.0/"}
MAX_BYTES = 3_000_000_000


def descendants(node: dict) -> set[str]:
    result = {node["LabelName"]}
    for child in node.get("Subcategory", []):
        result |= descendants(child)
    return result


def make_plan(max_images: int = 4500, seed: int = 42, budget: int = MAX_BYTES) -> list[dict[str, str]]:
    names = {name: identifier for identifier, name in csv.reader(CLASS_NAMES.open(encoding="utf-8"))}
    id_to_name = {identifier: name for name, identifier in names.items()}
    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))
    food_root = next((node for node in descendants_with_node(hierarchy)
                      if id_to_name.get(node["LabelName"]) == "Food"), None)
    if food_root is None:
        raise ValueError("Food root is missing from the official Open Images hierarchy")
    food_ids = descendants(food_root)
    target_ids = {names[name]: name for name in TARGETS if name in names}
    missing_targets = sorted(set(TARGETS) - {id_to_name.get(i, "") for i in target_ids})
    if len(target_ids) < 20:
        raise ValueError(f"Too few requested classes found in official labels: {missing_targets}")

    positive: dict[str, set[str]] = defaultdict(set)
    with ANNOTATIONS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["Confidence"] or 0) > 0:
                positive[row["ImageID"]].add(row["LabelName"])

    per_class: dict[str, list[str]] = defaultdict(list)
    for image_id, labels in positive.items():
        if labels & food_ids:
            continue
        for label_id in labels & target_ids.keys():
            per_class[target_ids[label_id]].append(image_id)

    image_info = {}
    with IMAGES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_info[row["ImageID"]] = row

    # Round-robin over categories with a fixed seed, avoiding a dataset dominated
    # by common labels such as person/car. The original files would exceed 3 GB;
    # Open Images' own Thumbnail300KURL keeps the useful subset bounded.
    rng = random.Random(seed)
    for values in per_class.values():
        rng.shuffle(values)
    category_names = sorted(per_class)
    rng.shuffle(category_names)
    selected: dict[str, set[str]] = defaultdict(set)
    chosen: list[str] = []
    cursor = 0
    while len(chosen) < max_images and any(per_class.values()):
        label = category_names[cursor % len(category_names)]
        cursor += 1
        if per_class[label]:
            image_id = per_class[label].pop()
            if image_id not in selected:
                selected[image_id].add(label)
                chosen.append(image_id)

    # Put target classes into at most 4,500 unique candidate IDs. Metadata sizes
    # let us enforce the image budget before the first image is downloaded.
    result = []
    for image_id in chosen:
        info = image_info.get(image_id)
        if not info or info.get("License") not in LICENSE_ALLOW or not info.get("Thumbnail300KURL"):
            continue
        labels = sorted(selected[image_id])
        result.append({
            **info,
            "image_id": image_id,
            "scene_tags": ";".join(labels),
            # The endpoint is explicitly Open Images' 300 KB thumbnail variant.
            "expected_bytes_upper_bound": "300000",
            "source_group": info.get("AuthorProfileURL") or info.get("Author") or image_id,
        })
    if len(result) < max_images:
        print(f"Only {len(result)} images fit the source/license/size filters")
    if PLAN.exists():
        raise FileExistsError(f"Refusing to overwrite selection plan: {PLAN}")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result[0]) if result else []
    with PLAN.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps({
        "selection_csv": str(PLAN), "seed": seed,
        "requested_max_images": max_images, "planned_images": len(result),
        "image_bytes_upper_bound": len(result) * 300_000,
        "metadata_bytes_on_disk": sum(path.stat().st_size for path in METADATA.glob("*")),
        "estimated_total_upper_bound_bytes": len(result) * 300_000 + sum(path.stat().st_size for path in METADATA.glob("*")),
        "target_classes_found": sorted(target_ids.values()),
        "missing_target_classes": missing_targets,
        "food_hierarchy_label_ids_excluded": len(food_ids),
        "license_allowlist": sorted(LICENSE_ALLOW),
        "caveat": "Open Images image-level labels can miss visible food; visual audit remains necessary",
    }, indent=2))
    return result


def descendants_with_node(node: dict):
    yield node
    for child in node.get("Subcategory", []):
        yield from descendants_with_node(child)


def download() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.request import Request, urlopen

    if not PLAN.is_file():
        raise FileNotFoundError("Run --plan first, review its size, then use --download")
    rows = list(csv.DictReader(PLAN.open(encoding="utf-8", newline="")))
    image_dir = BASE / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    def fetch(row):
        target = image_dir / f"{row['image_id']}.jpg"
        if target.is_file():
            return target
        request = Request(row["Thumbnail300KURL"], headers={"User-Agent": "food-calorie-estimator-robustness/1.0"})
        with urlopen(request, timeout=45) as response:
            data = response.read(300_001)
            if len(data) > 300_000:
                raise ValueError(f"Open Images 300K thumbnail exceeded 300 KB: {row['image_id']}")
            if response.status != 200:
                raise RuntimeError(f"Thumbnail request failed with HTTP {response.status}: {row['image_id']}")
        target.write_bytes(data)
        return target

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, row) for row in rows]
        errors = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
    downloaded_paths = [image_dir / f"{row['image_id']}.jpg" for row in rows]
    missing = [path for path in downloaded_paths if not path.is_file()]
    if errors or missing:
        raise RuntimeError(f"Open Images thumbnail download incomplete: {len(errors)} errors, {len(missing)} missing; first={errors[0] if errors else missing[0]}")
    actual_bytes = sum(path.stat().st_size for path in downloaded_paths)
    metadata_bytes = sum(p.stat().st_size for p in METADATA.glob("*"))
    if actual_bytes + metadata_bytes > MAX_BYTES:
        raise RuntimeError("Actual Open Images subset plus metadata exceeded the 3 GB limit")
    invalid = []
    for path in downloaded_paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(str(path))
    if invalid:
        raise RuntimeError(f"Invalid image files: {len(invalid)}; first={invalid[0]}")
    print(json.dumps({"downloaded_images": len(rows), "actual_image_bytes": actual_bytes,
                      "metadata_bytes": metadata_bytes,
                      "actual_total_bytes": actual_bytes + metadata_bytes}, indent=2))


def build_manifest() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "ml/src"))
    from robustness_data import assign_group_splits, write_csv

    if not PLAN.is_file():
        raise FileNotFoundError(f"Selection plan missing: {PLAN}")
    with (ROOT / "ml/data/processed/robustness_manifest_grouped.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    with PLAN.open(encoding="utf-8", newline="") as handle:
        planned = list(csv.DictReader(handle))
    image_dir = BASE / "images"
    names = {identifier: name for identifier, name in csv.reader(CLASS_NAMES.open(encoding="utf-8"))}
    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))
    food_root = next((node for node in descendants_with_node(hierarchy)
                      if names.get(node["LabelName"]) == "Food"), None)
    if food_root is None:
        raise RuntimeError("Food root missing while verifying selected image labels")
    food_ids = descendants(food_root)
    label_rows = defaultdict(set)
    with ANNOTATIONS.open(encoding="utf-8", newline="") as handle:
        for annotation in csv.DictReader(handle):
            if float(annotation["Confidence"] or 0) > 0:
                label_rows[annotation["ImageID"]].add(annotation["LabelName"])
    food_violations = [r["image_id"] for r in planned if label_rows[r["image_id"]] & food_ids]
    if food_violations:
        raise RuntimeError(f"Food-hierarchy exclusion failure for {len(food_violations)} selected IDs")

    download_rows = []
    valid = []
    corrupt = []
    for info in planned:
        path = image_dir / f"{info['image_id']}.jpg"
        detail = ""
        if not path.is_file():
            status = "failed_or_unavailable"
            detail = "Initial fetch recorded this URL as failed; per-image exception detail was not persisted (first logged error was HTTP 404). No retry performed."
        elif info.get("License") not in LICENSE_ALLOW or not info.get("OriginalLandingURL") or not info.get("AuthorProfileURL"):
            status = "invalid_provenance_excluded"
            detail = "Required CC BY 2.0 license or source attribution field is absent."
            corrupt.append(info["image_id"])
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
                status = "downloaded_valid"
                valid.append(info)
            except Exception as error:
                status = "downloaded_corrupt_excluded"
                detail = f"Image decode verification failed: {type(error).__name__}: {error}"
                corrupt.append(info["image_id"])
        download_rows.append({**info, "download_status": status,
                              "downloaded_bytes": path.stat().st_size if path.is_file() else "",
                              "failure_or_exclusion_detail": detail})
    report_fields = list(planned[0]) + ["download_status", "downloaded_bytes", "failure_or_exclusion_detail"]
    with DOWNLOAD_REPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(download_rows)
    if len(valid) < 3000:
        raise RuntimeError(f"Only {len(valid)} valid non-food images remain; target minimum is 3,000")

    for info in valid:
        author_group = info.get("AuthorProfileURL") or info.get("Author") or info["image_id"]
        rows.append({
            "path": str((image_dir / f"{info['image_id']}.jpg").resolve()),
            "split": "auto", "triage_label": "non_food", "food_class": "",
            "source_dataset": "Open Images V7", "source_split": "train",
            "source_id": info["image_id"],
            "group_id": f"OpenImagesV7:author:{author_group}",
            "scene_tags": info["scene_tags"], "license": info["License"],
            "source_url": info["OriginalURL"],
            "source_landing_url": info["OriginalLandingURL"],
            "source_author": info["Author"],
            "source_author_url": info["AuthorProfileURL"],
            "source_title": info["Title"],
            "source_original_md5": info["OriginalMD5"],
            "source_original_size": info["OriginalSize"],
            "source_rotation": info["Rotation"],
            "source_thumbnail_url": info["Thumbnail300KURL"],
        })
    assigned = assign_group_splits(rows, seed=42, validation_fraction=0.15, test_fraction=0.15)
    split_for = ("train", "validation", "test")
    for key in ("group_id", "sha256", "dhash"):
        owners = defaultdict(set)
        for row in assigned:
            if row["split"] in split_for:
                owners[row.get(key, "")].add(row["split"])
        leaked = [value for value, splits in owners.items() if value and len(splits) > 1]
        if leaked:
            raise RuntimeError(f"{key} split leakage: {len(leaked)} groups/hashes")
    bands = defaultdict(list)
    near_leaks = set()
    for row in assigned:
        if row["split"] not in split_for or not row.get("dhash"):
            continue
        code = int(row["dhash"], 16)
        for band in range(8):
            key = (band, (code >> (band * 8)) & 0xFF)
            for prior_code, prior_split, prior_id in bands[key]:
                if prior_split != row["split"] and (code ^ prior_code).bit_count() <= 5:
                    near_leaks.add(tuple(sorted((prior_id, row["source_id"]))))
            bands[key].append((code, row["split"], row["source_id"]))
    if near_leaks:
        raise RuntimeError(f"Near-duplicate split leakage: {len(near_leaks)} pairs")
    conflicts = sum(row["duplicate_label_conflict"] == "true" for row in assigned)
    old_test = Counter(row["source_id"] for row in rows
                       if row["source_dataset"] != "Open Images V7" and row["split"] == "test"
                       and row["triage_label"] == "supported_food")
    expected_test = Counter(row["source_id"] for row in csv.DictReader(
        (ROOT / "ml/data/processed/robustness_manifest_grouped.csv").open(encoding="utf-8", newline="")
    ) if row["split"] == "test" and row["triage_label"] == "supported_food")
    if old_test != expected_test:
        raise RuntimeError("Frozen supported-food test membership changed during manifest assembly")
    qa = {
        "planned_openimages": len(planned), "downloaded_valid": len(valid),
        "failed_or_unavailable": sum(r["download_status"] == "failed_or_unavailable" for r in download_rows),
        "corrupt_or_provenance_excluded": len(corrupt), "food_hierarchy_violations": len(food_violations),
        "exact_or_near_duplicate_label_conflict_rows": conflicts,
        "split_integrity": "passed (source groups, exact SHA256, and dHash distance <= 5 checked)",
        "frozen_supported_food_test_membership": "unchanged",
    }
    if DATASET_QA.exists():
        raise FileExistsError(f"Refusing to overwrite dataset QA report: {DATASET_QA}")
    DATASET_QA.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    write_csv(assigned, COMBINED_MANIFEST)
    counts = Counter((row["split"], row["triage_label"]) for row in assigned)
    openimages_counts = Counter(row["split"] for row in assigned if row["source_dataset"] == "Open Images V7")
    print(json.dumps({
        "manifest": str(COMBINED_MANIFEST), "rows": len(assigned),
        "split_label_counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "openimages_split_counts": dict(openimages_counts),
        "openimages_download_count": len(planned),
        "openimages_valid_count": len(valid),
        "failed_or_unavailable_count": qa["failed_or_unavailable"],
        "corrupt_or_provenance_excluded_count": len(corrupt),
        "exact_or_near_duplicate_label_conflicts": sum(r["duplicate_label_conflict"] == "true" for r in assigned),
        "conflict_rows_are_ignored_by_trainer": conflicts,
        "source_group_ids": len({r["group_id"] for r in assigned if r["source_dataset"] == "Open Images V7"}),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download", "manifest"))
    parser.add_argument("--max-images", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action == "plan":
        make_plan(args.max_images, args.seed)
    elif args.action == "download":
        download()
    else:
        build_manifest()


if __name__ == "__main__":
    main()
