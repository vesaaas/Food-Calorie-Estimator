"""Validation-only calibration of decision policies for the Phase 9.3 model.

The manifest reader deliberately selects validation rows only. Calibration and
selection cohorts are disjoint by source group. No test image paths are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "ml/experiments"
sys.path.insert(0, str(EXPERIMENTS))
import train_phase9_3_14class as runner  # noqa: E402


DEFAULT_CONFIG = EXPERIMENTS / "phase9_3_14class_config.json"
DEFAULT_OUTPUT = ROOT / "ml/artifacts/experiments/phase9_3_14class_policy_calibration"
SUPPORTED = "supported_food"
NON_FOOD = "non_food"
UNSUPPORTED = "unsupported_food"
SPLIT_SEED = "phase9_3_policy_calibration_v1"
TARGETS = {
    "supported_food_acceptance_overall_min": 0.97,
    "supported_food_acceptance_each_class_min": 0.95,
    "non_food_rejection_min": 0.80,
    "unsupported_food_rejection_min": 0.50,
    "supported_classification_accuracy_max_drop_absolute": 0.02,
    "source": "Project engineering targets, not reviewer-specified requirements.",
}


def load_validation_rows(manifest_path: Path, classes: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    supported_classes = set(classes[:-1])
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # Skip all non-validation rows before inspecting their paths. In
            # particular, no held-out test image path or content is touched.
            if row.get("split") != "validation":
                continue
            if row.get("duplicate_label_conflict", "false").lower() == "true":
                continue
            triage = row.get("triage_label")
            if triage == SUPPORTED:
                if row.get("food_class") not in supported_classes:
                    raise ValueError(f"Unknown supported class: {row.get('food_class')!r}")
            elif triage == NON_FOOD:
                if row.get("source_dataset") != "Open Images V7":
                    raise ValueError("non_food validation rows must be from Open Images V7")
            elif triage == UNSUPPORTED:
                if not row.get("food_class"):
                    raise ValueError("Unsupported-food rows must retain their own food class metadata")
            else:
                raise ValueError(f"Unexpected validation triage label: {triage!r}")
            if not row.get("group_id"):
                raise ValueError("Validation row is missing its source group id")
            path = Path(row["path"]).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Missing validation image: {path}")
            row["path"] = str(path.resolve())
            rows.append(row)
    groups = {row["triage_label"] for row in rows}
    if groups != {SUPPORTED, NON_FOOD, UNSUPPORTED}:
        raise ValueError(f"Validation is missing an evaluation group: {sorted(groups)}")
    for label in classes[:-1]:
        if not any(row["triage_label"] == SUPPORTED and row["food_class"] == label for row in rows):
            raise ValueError(f"Validation is missing supported class {label}")
    return rows


def split_calibration_selection(
    rows: list[dict[str, str]], *, seed: str = SPLIT_SEED,
) -> tuple[list[int], list[int], dict]:
    """Deterministically assign entire source groups to 50/50 cohorts."""
    assignment = {}
    for row in rows:
        group = row["group_id"]
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        assignment[group] = "calibration" if digest[0] < 128 else "selection"
    calibration = [i for i, row in enumerate(rows) if assignment[row["group_id"]] == "calibration"]
    selection = [i for i, row in enumerate(rows) if assignment[row["group_id"]] == "selection"]
    calibration_groups = {rows[i]["group_id"] for i in calibration}
    selection_groups = {rows[i]["group_id"] for i in selection}
    if calibration_groups & selection_groups:
        raise ValueError("Source groups overlap between calibration and selection")
    if not calibration or not selection:
        raise ValueError("Deterministic group split produced an empty cohort")
    for indices, name in ((calibration, "calibration"), (selection, "selection")):
        group_labels = {rows[i]["triage_label"] for i in indices}
        if group_labels != {SUPPORTED, NON_FOOD, UNSUPPORTED}:
            raise ValueError(f"{name} cohort is missing an evaluation group")
    return calibration, selection, {
        "method": "SHA256(seed + ':' + group_id), first digest byte <128 => calibration; otherwise selection",
        "seed": seed,
        "calibration_rows": len(calibration),
        "selection_rows": len(selection),
        "calibration_unique_groups": len(calibration_groups),
        "selection_unique_groups": len(selection_groups),
        "group_overlap_count": len(calibration_groups & selection_groups),
        "calibration_counts": _group_counts(rows, calibration),
        "selection_counts": _group_counts(rows, selection),
    }


def _group_counts(rows: list[dict[str, str]], indices: list[int]) -> dict:
    triage_counts = {name: 0 for name in (SUPPORTED, NON_FOOD, UNSUPPORTED)}
    supported_classes = {}
    for index in indices:
        row = rows[index]
        triage_counts[row["triage_label"]] += 1
        if row["triage_label"] == SUPPORTED:
            supported_classes[row["food_class"]] = supported_classes.get(row["food_class"], 0) + 1
    return {
        "triage": triage_counts,
        "supported_by_class": dict(sorted(supported_classes.items())),
    }


def _policy_acceptance(probabilities: np.ndarray, policy: dict) -> np.ndarray:
    food_probabilities = probabilities[:, :-1]
    winning_food = food_probabilities.argmax(axis=1)
    score = food_probabilities.max(axis=1)
    kind = policy["kind"]
    if kind == "argmax":
        return probabilities.argmax(axis=1) != probabilities.shape[1] - 1
    if kind == "max_supported_probability":
        return score >= float(policy["threshold"])
    if kind == "prediction_margin":
        margin = score - probabilities[:, -1]
        return margin >= float(policy["threshold"])
    if kind == "class_conditioned_max_supported_probability":
        thresholds = np.asarray(policy["thresholds"], dtype=np.float64)
        if thresholds.shape != (food_probabilities.shape[1],):
            raise ValueError("Class-conditioned policy requires one threshold per food class")
        return score >= thresholds[winning_food]
    raise ValueError(f"Unknown decision policy: {kind}")


def evaluate_policy(
    rows: list[dict[str, str]], probabilities: np.ndarray,
    production_probabilities: np.ndarray, classes: tuple[str, ...], policy: dict,
) -> dict:
    if probabilities.shape != (len(rows), len(classes)):
        raise ValueError(f"Unexpected candidate probability shape: {probabilities.shape}")
    if production_probabilities.shape != (len(rows), len(classes) - 1):
        raise ValueError("Production probability rows/labels do not match validation rows")
    if not np.isfinite(probabilities).all() or not np.isfinite(production_probabilities).all():
        raise ValueError("Validation probabilities contain NaN or Infinity")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Candidate probabilities are outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("14-class output rows do not sum to 1")

    accepted = _policy_acceptance(probabilities, policy)
    predicted_supported = probabilities[:, :-1].argmax(axis=1)
    raw_top1 = probabilities.argmax(axis=1)
    production_top1 = production_probabilities.argmax(axis=1)
    supported = np.asarray([row["triage_label"] == SUPPORTED for row in rows])
    non_food = np.asarray([row["triage_label"] == NON_FOOD for row in rows])
    unsupported = np.asarray([row["triage_label"] == UNSUPPORTED for row in rows])
    truth = np.asarray([
        classes.index(row["food_class"]) if row["triage_label"] == SUPPORTED else -1
        for row in rows
    ], dtype=np.int64)
    if not supported.any() or not non_food.any() or not unsupported.any():
        raise ValueError("Evaluation needs supported, non_food, and unsupported_food rows")

    per_class = {}
    for class_index, label in enumerate(classes[:-1]):
        mask = supported & (truth == class_index)
        if not mask.any():
            raise ValueError(f"Selection cohort lacks supported class {label}")
        per_class[label] = {
            "samples": int(mask.sum()),
            "acceptance": float(accepted[mask].mean()),
            "raw_14class_top1_accuracy": float((raw_top1[mask] == class_index).mean()),
            "effective_accepted_and_correct": float(
                (accepted[mask] & (predicted_supported[mask] == class_index)).mean()
            ),
        }

    supported_top1_accuracy = float((raw_top1[supported] == truth[supported]).mean())
    effective_accuracy = float(
        (accepted[supported] & (predicted_supported[supported] == truth[supported])).mean()
    )
    production_accuracy = float((production_top1[supported] == truth[supported]).mean())
    accuracy_drop = production_accuracy - effective_accuracy
    overall_acceptance = float(accepted[supported].mean())
    class_floor = min(item["acceptance"] for item in per_class.values())
    non_food_rejection = float((~accepted[non_food]).mean())
    unsupported_rejection = float((~accepted[unsupported]).mean())
    gate = {
        "supported_food_acceptance_overall": overall_acceptance,
        "supported_food_acceptance_each_class_min": class_floor,
        "non_food_rejection": non_food_rejection,
        "unsupported_food_rejection": unsupported_rejection,
        "supported_food_raw_14class_top1_accuracy": supported_top1_accuracy,
        "effective_accepted_and_correct_classification_accuracy": effective_accuracy,
        "production_13class_top1_accuracy_same_rows": production_accuracy,
        "effective_classification_accuracy_drop_absolute": accuracy_drop,
    }
    gate["meets_targets"] = bool(
        overall_acceptance >= TARGETS["supported_food_acceptance_overall_min"]
        and class_floor >= TARGETS["supported_food_acceptance_each_class_min"]
        and non_food_rejection >= TARGETS["non_food_rejection_min"]
        and unsupported_rejection >= TARGETS["unsupported_food_rejection_min"]
        and accuracy_drop <= TARGETS["supported_classification_accuracy_max_drop_absolute"]
    )
    return {
        "policy": policy,
        "decision_metrics": gate,
        "supported_food_by_class": per_class,
        "samples": {
            "supported_food": int(supported.sum()),
            "non_food": int(non_food.sum()),
            "unsupported_food_evaluation_only": int(unsupported.sum()),
        },
        "confusion_by_group": {
            "supported_food": {"accepted": int((accepted & supported).sum()),
                                "rejected": int((~accepted & supported).sum())},
            "non_food": {"accepted_as_food": int((accepted & non_food).sum()),
                         "rejected_as_non_food": int((~accepted & non_food).sum())},
            "unsupported_food_evaluation_only": {
                "accepted_as_supported_food": int((accepted & unsupported).sum()),
                "rejected_by_policy": int((~accepted & unsupported).sum()),
            },
        },
    }


def _target_attainment(metrics: dict) -> float:
    item = metrics["decision_metrics"]
    ratios = [
        min(item["supported_food_acceptance_overall"] / TARGETS["supported_food_acceptance_overall_min"], 1.0),
        min(item["supported_food_acceptance_each_class_min"] / TARGETS["supported_food_acceptance_each_class_min"], 1.0),
        min(item["non_food_rejection"] / TARGETS["non_food_rejection_min"], 1.0),
        min(item["unsupported_food_rejection"] / TARGETS["unsupported_food_rejection_min"], 1.0),
        1.0 if item["effective_classification_accuracy_drop_absolute"]
        <= TARGETS["supported_classification_accuracy_max_drop_absolute"]
        else max(0.0, 1.0 - (item["effective_classification_accuracy_drop_absolute"]
                             - TARGETS["supported_classification_accuracy_max_drop_absolute"]) / 0.02),
    ]
    return float(np.mean(ratios))


def _calibration_key(metrics: dict, threshold: float) -> tuple:
    gate = metrics["decision_metrics"]
    return (
        gate["meets_targets"],
        _target_attainment(metrics),
        gate["supported_food_acceptance_overall"],
        gate["supported_food_acceptance_each_class_min"],
        gate["non_food_rejection"],
        gate["unsupported_food_rejection"],
        gate["effective_accepted_and_correct_classification_accuracy"],
        threshold,
    )


def _choose_global_threshold(
    rows: list[dict[str, str]], candidate_probs: np.ndarray, production_probs: np.ndarray,
    classes: tuple[str, ...], *, kind: str,
) -> tuple[dict, dict]:
    if kind == "max_supported_probability":
        scores = candidate_probs[:, :-1].max(axis=1)
        name = "max_supported_class_probability"
    elif kind == "prediction_margin":
        scores = candidate_probs[:, :-1].max(axis=1) - candidate_probs[:, -1]
        name = "max_supported_probability_minus_non_food_probability"
    else:
        raise ValueError(f"Unsupported global score: {kind}")
    candidates = np.unique(np.concatenate((scores, [np.nextafter(scores.max(), np.inf)])))
    best_policy, best_metrics, best_key = None, None, None
    for threshold in candidates:
        policy = {"kind": kind, "score": name, "threshold": float(threshold)}
        result = evaluate_policy(rows, candidate_probs, production_probs, classes, policy)
        key = _calibration_key(result, float(threshold))
        if best_key is None or key > best_key:
            best_policy, best_metrics, best_key = policy, result, key
    return best_policy, best_metrics


def _choose_class_conditioned_thresholds(
    rows: list[dict[str, str]], candidate_probs: np.ndarray, production_probs: np.ndarray,
    classes: tuple[str, ...], initial_threshold: float,
) -> tuple[dict, dict]:
    food_probs = candidate_probs[:, :-1]
    winning_class = food_probs.argmax(axis=1)
    winning_score = food_probs.max(axis=1)
    thresholds = np.full(len(classes) - 1, initial_threshold, dtype=np.float64)
    policy = {
        "kind": "class_conditioned_max_supported_probability",
        "score": "probability of the highest-probability supported-food output; threshold is indexed by that predicted food class",
        "thresholds": thresholds.tolist(),
        "selection_method": "two deterministic coordinate-ascent passes over calibration rows only; objective prioritizes target feasibility, mean capped target attainment, then supported retention/non-food rejection/unsupported rejection",
    }
    best_metrics = evaluate_policy(rows, candidate_probs, production_probs, classes, policy)
    best_key = _calibration_key(best_metrics, initial_threshold)
    for _ in range(2):
        for class_index in range(len(classes) - 1):
            class_scores = winning_score[winning_class == class_index]
            if not len(class_scores):
                continue
            candidates = np.unique(np.concatenate((
                class_scores,
                [np.nextafter(class_scores.max(), np.inf)],
            )))
            local_best = None
            local_key = None
            for threshold in candidates:
                thresholds[class_index] = float(threshold)
                policy["thresholds"] = thresholds.tolist()
                result = evaluate_policy(rows, candidate_probs, production_probs, classes, policy)
                key = _calibration_key(result, float(threshold))
                if local_key is None or key > local_key:
                    local_best, local_key = (float(threshold), result), key
            thresholds[class_index] = local_best[0]
            best_metrics = local_best[1]
            policy["thresholds"] = thresholds.tolist()
            best_key = _calibration_key(best_metrics, float(np.min(thresholds)))
    policy["thresholds"] = thresholds.tolist()
    best_metrics = evaluate_policy(rows, candidate_probs, production_probs, classes, policy)
    return policy, best_metrics


def calibrate_policies(
    rows: list[dict[str, str]], calibration_indices: list[int],
    candidate_probs: np.ndarray, production_probs: np.ndarray,
    classes: tuple[str, ...],
) -> tuple[dict[str, dict], dict[str, dict]]:
    cal_rows = [rows[i] for i in calibration_indices]
    cal_candidate = candidate_probs[calibration_indices]
    cal_production = production_probs[calibration_indices]
    argmax_policy = {"kind": "argmax", "rule": "accept iff 14-class argmax is one of the first 13 food labels"}
    results = {
        "argmax_baseline": (
            argmax_policy,
            evaluate_policy(cal_rows, cal_candidate, cal_production, classes, argmax_policy),
        ),
    }
    for kind in ("max_supported_probability", "prediction_margin"):
        policy, result = _choose_global_threshold(
            cal_rows, cal_candidate, cal_production, classes, kind=kind,
        )
        results[kind] = (policy, result)
    conditioned_policy, conditioned_metrics = _choose_class_conditioned_thresholds(
        cal_rows, cal_candidate, cal_production, classes,
        initial_threshold=results["max_supported_probability"][0]["threshold"],
    )
    results["class_conditioned_max_supported_probability"] = (
        conditioned_policy, conditioned_metrics,
    )

    metadata = {}
    for name, (policy, result) in results.items():
        metadata[name] = {
            "policy": policy,
            "calibration_metrics": result,
            "calibration_target_attainment": _target_attainment(result),
            "calibration_targets_met": result["decision_metrics"]["meets_targets"],
        }
    return results, metadata


def _rank_selection_policy(result: dict) -> tuple:
    gate = result["decision_metrics"]
    return (
        gate["meets_targets"],
        _target_attainment(result),
        gate["supported_food_acceptance_overall"],
        gate["supported_food_acceptance_each_class_min"],
        gate["non_food_rejection"],
        gate["unsupported_food_rejection"],
        gate["effective_accepted_and_correct_classification_accuracy"],
    )


def _failure_examples(
    rows: list[dict[str, str]], probs: np.ndarray, classes: tuple[str, ...], policy: dict,
    group: str, *, accepted_errors: bool, limit: int = 8,
) -> list[dict]:
    accepted = _policy_acceptance(probs, policy)
    food_probs = probs[:, :-1]
    winning = food_probs.argmax(axis=1)
    score = food_probs.max(axis=1)
    chosen = [
        i for i, row in enumerate(rows)
        if row["triage_label"] == group and bool(accepted[i]) == accepted_errors
    ]
    chosen.sort(key=lambda i: float(score[i]), reverse=accepted_errors)
    return [{
        "source_id": rows[i].get("source_id"),
        "source_dataset": rows[i].get("source_dataset"),
        "food_class": rows[i].get("food_class"),
        "scene_tags": rows[i].get("scene_tags"),
        "predicted_supported_class": classes[int(winning[i])],
        "max_supported_probability": float(score[i]),
        "non_food_probability": float(probs[i, -1]),
        "path": rows[i]["path"],
    } for i in chosen[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = runner.load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    experiment_root = (ROOT / "ml/artifacts/experiments").resolve()
    try:
        output_dir.relative_to(experiment_root)
    except ValueError as error:
        parser.error(f"Output directory must be under {experiment_root}: {error}")
    if output_dir.exists():
        parser.error(f"Refusing to overwrite policy calibration output: {output_dir}")

    classes = tuple(config["classes"])
    manifest = ROOT / config["manifest"]
    fine_checkpoint = ROOT / "ml/artifacts/experiments/phase9_3_14class_finetune/phase9_3_14class_finetune_best.keras"
    production_checkpoint = ROOT / config["initial_checkpoint"]
    for path in (manifest, fine_checkpoint, production_checkpoint):
        if not path.is_file():
            parser.error(f"Required validation input is missing: {path}")

    rows = load_validation_rows(manifest, classes)
    calibration_indices, selection_indices, partition_report = split_calibration_selection(rows)
    fine_model = tf.keras.models.load_model(fine_checkpoint, compile=False)
    production_model = tf.keras.models.load_model(production_checkpoint, compile=False)
    if tuple(fine_model.output_shape) != (None, len(classes)):
        parser.error(f"Fine-tuned checkpoint has unexpected output shape: {fine_model.output_shape}")
    if tuple(production_model.output_shape) != (None, len(classes) - 1):
        parser.error(f"Production checkpoint has unexpected output shape: {production_model.output_shape}")

    candidate_probabilities = runner.predict_rows(fine_model, rows)
    production_probabilities = runner.predict_rows(production_model, rows)
    selected_policies, calibration_report = calibrate_policies(
        rows, calibration_indices, candidate_probabilities,
        production_probabilities, classes,
    )

    selection_rows = [rows[i] for i in selection_indices]
    selection_candidate = candidate_probabilities[selection_indices]
    selection_production = production_probabilities[selection_indices]
    selection_report = {}
    for name, (policy, _) in selected_policies.items():
        result = evaluate_policy(
            selection_rows, selection_candidate, selection_production, classes, policy,
        )
        result["target_attainment"] = _target_attainment(result)
        result["targets_met"] = result["decision_metrics"]["meets_targets"]
        selection_report[name] = result

    # Freeze the candidate policy family using calibration rows only. The
    # selection cohort is reserved for the final comparison and examples.
    chosen_name = max(
        selected_policies,
        key=lambda name: _rank_selection_policy(
            selected_policies[name][1]
        ),
    )
    chosen_policy = selected_policies[chosen_name][0]
    chosen_selection = selection_report[chosen_name]
    confusion_examples = {
        "unsupported_food_false_accepts": _failure_examples(
            selection_rows, selection_candidate, classes, chosen_policy,
            UNSUPPORTED, accepted_errors=True,
        ),
        "non_food_false_accepts": _failure_examples(
            selection_rows, selection_candidate, classes, chosen_policy,
            NON_FOOD, accepted_errors=True,
        ),
        "supported_food_false_rejections": _failure_examples(
            selection_rows, selection_candidate, classes, chosen_policy,
            SUPPORTED, accepted_errors=False,
        ),
    }
    all_group_ids = {row["group_id"] for row in rows}
    report = {
        "experiment": "phase9_3_14class_validation_policy_calibration",
        "status": "all_targets_met_on_selection" if chosen_selection["targets_met"]
        else "no_policy_met_all_targets_on_selection",
        "manifest": str(manifest.resolve()),
        "fine_tuned_checkpoint": str(fine_checkpoint.resolve()),
        "production_baseline_checkpoint": str(production_checkpoint.resolve()),
        "classes": list(classes),
        "targets": TARGETS,
        "cohort_split": partition_report,
        "validation_integrity": {
            "rows_read_from_validation_split_only": len(rows),
            "source_group_count": len(all_group_ids),
            "calibration_selection_group_overlap": 0,
            "unsupported_food_kept_separate_from_non_food": True,
            "held_out_test_images_read_or_evaluated": False,
        },
        "calibration_policy_selection": calibration_report,
        "untouched_selection_performance": selection_report,
        "best_conservative_policy": {
            "name": chosen_name,
            "policy": chosen_policy,
            "selection_metrics": chosen_selection,
            "selection_basis": "Policy family and thresholds frozen using calibration rows only; selection rows were not used to choose or tune the policy.",
            "calibration_ranking": "Prefer a policy meeting all predeclared targets; otherwise maximize mean capped target attainment, then supported-food retention, per-class floor, non-food rejection, unsupported-food rejection, and effective accepted-and-correct accuracy.",
        },
        "confusion_examples": confusion_examples,
        "production_assets_modified": False,
        "model_weights_modified": False,
        "tflite_export_performed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "policy_calibration_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "best_policy": chosen_name,
        "best_policy_parameters": chosen_policy,
        "selection_metrics": chosen_selection["decision_metrics"],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
