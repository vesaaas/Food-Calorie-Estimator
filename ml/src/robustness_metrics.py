"""Summarize per-image classifier reports without touching model artifacts.

Input JSON uses the `predictions` records emitted by evaluate_tflite_confidence.py.
The tool computes confusion, per-class recall, confidence slices, and ECE. Two
reports can be paired to compare candidates on matching image indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .dataset import PHASE5_CLASSES
else:
    from dataset import PHASE5_CLASSES


def _confidence(row: dict) -> float:
    return float(row.get("maximum_softmax_confidence", row.get("confidence")))


def summarize(report: dict, *, threshold: float = 0.55, bins: int = 15) -> dict:
    rows = report.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input report must contain nonempty per-image predictions")
    if not 0 <= threshold <= 1 or bins < 1:
        raise ValueError("threshold must be in [0,1] and bins must be positive")
    labels = {name: index for index, name in enumerate(PHASE5_CLASSES)}
    parsed = []
    for row in rows:
        true_name = row.get("true_class")
        pred_name = row.get("predicted_class")
        if true_name not in labels or pred_name not in labels:
            raise ValueError(f"Unknown class in prediction row: {row}")
        conf = _confidence(row)
        if not np.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("Confidence must be finite and in [0,1]")
        parsed.append((labels[true_name], labels[pred_name], conf))
    y_true = np.asarray([row[0] for row in parsed], dtype=np.int64)
    y_pred = np.asarray([row[1] for row in parsed], dtype=np.int64)
    conf = np.asarray([row[2] for row in parsed], dtype=np.float64)
    correct = y_true == y_pred
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    per_class = {}
    for index, name in enumerate(PHASE5_CLASSES):
        selected = y_true == index
        count = int(selected.sum())
        class_correct = correct[selected]
        class_conf = conf[selected]
        per_class[name] = {
            "samples": count,
            "top1_accuracy": float(class_correct.mean()) if count else None,
            "recall": float(class_correct.mean()) if count else None,
            "low_confidence_correct": int(np.sum(class_correct & (class_conf < threshold))),
            "low_confidence_correct_rate": (
                float(np.sum(class_correct & (class_conf < threshold)) / class_correct.sum())
                if class_correct.sum() else None
            ),
            "high_confidence_incorrect": int(np.sum(~class_correct & (class_conf >= threshold))),
            "confidence_correct": _describe(class_conf[class_correct]),
            "confidence_incorrect": _describe(class_conf[~class_correct]),
        }
    calibration = _calibration(y_true, y_pred, conf, bins)
    accepted = conf >= threshold
    return {
        "source_report": report.get("model"), "split": report.get("split"),
        "samples": len(parsed), "accuracy": float(correct.mean()),
        "threshold": threshold,
        "accepted_rate": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "correct_rejected": int(np.sum(correct & ~accepted)),
        "wrong_accepted": int(np.sum(~correct & accepted)),
        "confidence_correct": _describe(conf[correct]),
        "confidence_incorrect": _describe(conf[~correct]),
        "ece": calibration["ece"], "calibration_bins": calibration["bins"],
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(PHASE5_CLASSES), "counts_true_rows_predicted_columns": matrix.tolist(),
        },
        "sample_ids": [str(row.get("index", i)) for i, row in enumerate(rows)],
        "sample_labels": [row[0] for row in parsed],
        "sample_predictions": [row[1] for row in parsed],
    }


def _describe(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _calibration(y_true: np.ndarray, y_pred: np.ndarray,
                 confidence: np.ndarray, bins: int) -> dict:
    correct = y_true == y_pred
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1
            else confidence < edges[index + 1]
        )
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correct[selected].mean())
        average_confidence = float(confidence[selected].mean())
        ece += count / len(confidence) * abs(accuracy - average_confidence)
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": count, "accuracy": accuracy,
            "mean_confidence": average_confidence,
        })
    return {"ece": float(ece), "bins": rows}


def compare(left: dict, right: dict) -> dict:
    if left["sample_ids"] != right["sample_ids"] or left["sample_labels"] != right["sample_labels"]:
        raise ValueError("Candidate reports do not use the same ordered images and labels")
    if left["split"] != right["split"]:
        raise ValueError("Candidate reports have different split identifiers")
    return {
        "split": left["split"], "samples": left["samples"],
        "left_accuracy": left["accuracy"], "right_accuracy": right["accuracy"],
        "accuracy_delta_right_minus_left": right["accuracy"] - left["accuracy"],
        "per_class_accuracy_delta": {
            name: (
                right["per_class"][name]["top1_accuracy"] - left["per_class"][name]["top1_accuracy"]
                if right["per_class"][name]["top1_accuracy"] is not None
                and left["per_class"][name]["top1_accuracy"] is not None else None
            )
            for name in PHASE5_CLASSES
        },
        "left_ece": left["ece"], "right_ece": right["ece"],
    }


def _self_test() -> None:
    report = {"model": "synthetic", "split": "test", "predictions": [
        {"index": 0, "true_class": "pizza", "predicted_class": "pizza", "confidence": 0.53},
        {"index": 1, "true_class": "sushi", "predicted_class": "steak", "confidence": 0.9},
    ]}
    result = summarize(report)
    assert result["samples"] == 2 and result["correct_rejected"] == 1
    assert result["wrong_accepted"] == 1
    assert result["per_class"]["pizza"]["low_confidence_correct"] == 1
    assert result["confusion_matrix"]["counts_true_rows_predicted_columns"][2][3] == 1
    assert 0 <= result["ece"] <= 1
    compare(result, result)
    print("Robustness metrics self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.report is None:
        parser.error("report is required unless --self-test is used")
    left_payload = json.loads(args.report.read_text(encoding="utf-8"))
    if isinstance(left_payload.get("held_out_test"), dict):
        left_payload = {**left_payload["held_out_test"], "split": "test", "model": left_payload.get("checkpoint")}
    left = summarize(left_payload, threshold=args.threshold, bins=args.bins)
    if args.compare:
        right_payload = json.loads(args.compare.read_text(encoding="utf-8"))
        if isinstance(right_payload.get("held_out_test"), dict):
            right_payload = {**right_payload["held_out_test"], "split": "test", "model": right_payload.get("checkpoint")}
        right = summarize(right_payload, threshold=args.threshold, bins=args.bins)
        result = {"left": left, "right": right, "comparison": compare(left, right)}
    else:
        result = left
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
