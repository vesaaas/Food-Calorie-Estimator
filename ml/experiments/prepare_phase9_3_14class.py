"""Check Phase 9.3 14-class experiment inputs using metadata only.

No image is decoded, copied, modified, or used for model inference. This checks
the existing manifest/QA artifacts and prints a readiness summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "ml/experiments/phase9_3_14class_config.json"
VALID_SPLITS = {"train", "validation", "test"}
SUPPORTED = "supported_food"
UNSUPPORTED = "unsupported_food"
NON_FOOD = "non_food"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _membership_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row.get("source_dataset", ""), row.get("source_id", ""),
            row.get("food_class", ""), row.get("sha256", ""))


def _verify_split_integrity(rows: list[dict[str, str]]) -> dict[str, int]:
    owners: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in ("group_id", "sha256", "dhash")
    }
    bands: dict[tuple[int, int], list[tuple[str, int, str]]] = defaultdict(list)
    near_pairs = set()
    for row in rows:
        split = row["split"]
        source_id = row.get("source_id", "")
        for field in owners:
            value = row.get(field, "").strip()
            if not value:
                raise ValueError(f"Missing {field} in {split} row {source_id}")
            owners[field][value].add(split)

        code = int(row["dhash"], 16)
        for band in range(8):
            key = (band, (code >> (band * 8)) & 0xFF)
            # Any 64-bit dHash pair within five bits must share at least three
            # exact 8-bit bands, so this locality-sensitive index is complete
            # for the manifest's dHash-distance<=5 guarantee.
            for prior_split, prior_code, prior_id in bands[key]:
                if prior_split != split and bin(code ^ prior_code).count("1") <= 5:
                    near_pairs.add(tuple(sorted((prior_id, source_id))))
            bands[key].append((split, code, source_id))

    leaks = {
        field: sum(len(splits) > 1 for splits in values.values())
        for field, values in owners.items()
    }
    if any(leaks.values()):
        raise ValueError(f"Cross-split group/hash leakage: {leaks}")
    if near_pairs:
        raise ValueError(f"Cross-split dHash<=5 near-duplicate pairs: {len(near_pairs)}")
    return {**leaks, "near_duplicate_pairs": len(near_pairs)}


def check(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_classes = config["classes"]
    if len(expected_classes) != 14 or len(set(expected_classes)) != 14:
        raise ValueError("The experiment must have 14 unique labels")
    if config["class_to_idx"] != {name: index for index, name in enumerate(expected_classes)}:
        raise ValueError("Explicit class_to_idx mapping does not match class order")
    if expected_classes[-1] != NON_FOOD:
        raise ValueError("non_food must be the explicit 14th output label")

    manifest_path = ROOT / config["manifest"]
    reference_path = ROOT / config["supported_manifest_for_test_membership_check"]
    qa_path = ROOT / config["openimages_qa"]
    checkpoint_path = ROOT / config["initial_checkpoint"]
    checkpoint_metadata_path = ROOT / config["initial_checkpoint_metadata"]
    contract_path = ROOT / config["production_tflite_contract"]
    for required in (manifest_path, reference_path, qa_path, checkpoint_path,
                     checkpoint_metadata_path, contract_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    supported_classes = expected_classes[:-1]
    if checkpoint_metadata.get("classes") != supported_classes:
        raise ValueError("Checkpoint class order differs from the first 13 experiment labels")
    if checkpoint_metadata.get("class_to_idx") != {
        name: index for index, name in enumerate(supported_classes)
    }:
        raise ValueError("Checkpoint class indices differ from the experiment class order")
    if contract.get("labels") != supported_classes:
        raise ValueError("Production TFLite labels differ from the first 13 experiment labels")
    if contract.get("input", {}).get("tensor_shape") != [1, 224, 224, 3]:
        raise ValueError("Production TFLite input contract differs from 224x224 NHWC")
    if contract.get("input", {}).get("pixel_range") != [0, 255]:
        raise ValueError("Production TFLite input range is not raw [0,255]")

    rows_all = _read_csv(manifest_path)
    conflicts = [row for row in rows_all
                 if row.get("duplicate_label_conflict", "false").lower() == "true"]
    rows = [row for row in rows_all
            if row.get("split") in VALID_SPLITS
            and row.get("duplicate_label_conflict", "false").lower() != "true"]
    if not rows:
        raise ValueError("No non-conflict train/validation/test manifest records found")

    missing_files = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} manifest files missing; first={missing_files[0]}")

    counts = Counter((row["split"], row["triage_label"]) for row in rows)
    expected_counts = config["expected_manifest_counts_after_conflict_exclusion"]
    named_counts = {
        "train_supported_food": counts[("train", SUPPORTED)],
        "train_non_food": counts[("train", NON_FOOD)],
        "validation_supported_food": counts[("validation", SUPPORTED)],
        "validation_non_food": counts[("validation", NON_FOOD)],
        "validation_unsupported_food_evaluation_only": counts[("validation", UNSUPPORTED)],
        "test_supported_food_frozen": counts[("test", SUPPORTED)],
        "test_non_food": counts[("test", NON_FOOD)],
        "test_unsupported_food_evaluation_only": counts[("test", UNSUPPORTED)],
    }
    if named_counts != expected_counts:
        raise ValueError(f"Manifest split counts changed: expected={expected_counts}, actual={named_counts}")

    by_split_class = Counter(
        (row["split"], row["food_class"])
        for row in rows if row["triage_label"] == SUPPORTED
    )
    for split in VALID_SPLITS:
        missing = [name for name in expected_classes[:-1] if by_split_class[(split, name)] == 0]
        if missing:
            raise ValueError(f"{split} lacks supported classes: {missing}")

    bad_nonfood_sources = [row for row in rows if row["triage_label"] == NON_FOOD
                           and row.get("source_dataset") != "Open Images V7"]
    if bad_nonfood_sources:
        raise ValueError(f"Non-food class contains non-Open-Images rows: {len(bad_nonfood_sources)}")

    # Model-fitting rows include only the 14 explicit classes. Unsupported food
    # remains a separate validation/test rejection cohort and never becomes
    # the non_food target.
    train_rows = [row for row in rows if row["split"] == "train"]
    model_train_rows = [row for row in train_rows
                        if row["triage_label"] in (SUPPORTED, NON_FOOD)]
    if any(row["triage_label"] == UNSUPPORTED for row in model_train_rows):
        raise ValueError("Unsupported-food examples must not enter 14-class training")
    fit_class_counts = Counter(
        row["food_class"] if row["triage_label"] == SUPPORTED else NON_FOOD
        for row in model_train_rows
    )
    if set(fit_class_counts) != set(expected_classes):
        raise ValueError(f"Unexpected training labels: {sorted(fit_class_counts)}")

    # Confirm the 14-way manifest keeps the exact frozen supported-food test
    # membership of the existing grouped robustness manifest.
    reference_rows = [row for row in _read_csv(reference_path)
                      if row.get("split") == "test"
                      and row.get("triage_label") == SUPPORTED
                      and row.get("duplicate_label_conflict", "false").lower() != "true"]
    candidate_rows = [row for row in rows
                      if row.get("split") == "test" and row.get("triage_label") == SUPPORTED]
    reference_members = Counter(_membership_key(row) for row in reference_rows)
    candidate_members = Counter(_membership_key(row) for row in candidate_rows)
    if reference_members != candidate_members:
        raise ValueError("Frozen supported-food test membership differs from grouped manifest")

    integrity = _verify_split_integrity(rows)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa.get("split_integrity") != "passed (source groups, exact SHA256, and dHash distance <= 5 checked)":
        raise ValueError("Open Images QA does not record source/hash/near-duplicate split integrity")
    if qa.get("frozen_supported_food_test_membership") != "unchanged":
        raise ValueError("Open Images QA does not confirm frozen supported-food test membership")
    if qa.get("downloaded_valid") != 3672 or qa.get("food_hierarchy_violations") != 0:
        raise ValueError("Open Images QA counts/food-hierarchy check differ from expected local artifact")

    train_scene_tags = {
        tag for row in train_rows if row["triage_label"] == NON_FOOD
        for tag in row.get("scene_tags", "").split(";") if tag
    }
    if len(train_scene_tags) < 30:
        raise ValueError(f"Non-food training source appears too narrow: {len(train_scene_tags)} scene tags")

    return {
        "status": "ready_for_separate_experiment_training; no model training or image decoding performed",
        "config": str(config_path),
        "class_mapping": config["class_to_idx"],
        "train_examples_per_model_class": dict(sorted(fit_class_counts.items())),
        "evaluation_group_counts": named_counts,
        "supported_test_membership_matches_reference": len(candidate_rows),
        "cross_split_integrity": integrity,
        "nonfood_training_scene_tag_count": len(train_scene_tags),
        "conflict_rows_excluded": len(conflicts),
        "openimages_qa": {
            "valid_images": qa["downloaded_valid"],
            "food_hierarchy_violations": qa["food_hierarchy_violations"],
            "split_integrity": qa["split_integrity"],
        },
        "production_checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_class_order_matches": checkpoint_metadata["classes"] == supported_classes,
        "tflite_contract_labels_match": contract["labels"] == supported_classes,
        "production_outputs_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(check(args.config), indent=2))


if __name__ == "__main__":
    main()
