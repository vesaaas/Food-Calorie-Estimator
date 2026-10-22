"""Measure confidence-threshold behavior of the deployed MobileNetV3 TFLite model.

Run validation first. After selecting a threshold from validation, run the held-out
test split once with ``--split test --threshold VALUE``. No model is trained or
modified by this script.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .dataset import DATA_DIR, PHASE5_CLASSES
    from .download_phase5_dataset import DEFAULT_OUTPUT
    from .export_mobilenet_v3_tflite import DEFAULT_TFLITE
    from .mobilenet_v3_pipeline import TFLiteFoodClassifier
    from .tf_phase5_data import _model_input_from_torch_image, create_tf_phase5_datasets
else:
    from dataset import DATA_DIR, PHASE5_CLASSES
    from download_phase5_dataset import DEFAULT_OUTPUT
    from export_mobilenet_v3_tflite import DEFAULT_TFLITE
    from mobilenet_v3_pipeline import TFLiteFoodClassifier
    from tf_phase5_data import _model_input_from_torch_image, create_tf_phase5_datasets


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.30, 0.701, 0.05))
DEFAULT_VALIDATION_REPORT = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_confidence_validation.json"
DEFAULT_TEST_REPORT = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_confidence_test.json"


def confidence_statistics(predictions: list[dict]) -> dict:
    values = np.asarray(
        [row["confidence"] for row in predictions], dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "percentiles": {}, "min": None, "max": None}
    percentiles = (0, 10, 25, 50, 75, 90, 95, 99, 100)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "percentiles": {
            str(percentile): float(np.percentile(values, percentile))
            for percentile in percentiles
        },
        "min": float(values.min()),
        "max": float(values.max()),
    }


def threshold_metrics(predictions: list[dict], threshold: float) -> dict:
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    total = len(predictions)
    def confidence(row: dict) -> float:
        return row.get("maximum_softmax_confidence", row.get("confidence"))

    accepted = [row for row in predictions if confidence(row) >= threshold]
    rejected = [row for row in predictions if confidence(row) < threshold]
    accepted_correct = sum(row["correct"] for row in accepted)
    accepted_incorrect = len(accepted) - accepted_correct
    original_correct = sum(row["correct"] for row in predictions)
    correct_rejected = sum(row["correct"] for row in rejected)
    return {
        "threshold": float(threshold),
        "comparison": "accepted when maximum softmax confidence >= threshold",
        "total": total,
        "accepted": len(accepted),
        "accepted_percent": 100.0 * len(accepted) / total if total else None,
        "rejected": len(rejected),
        "rejected_percent": 100.0 * len(rejected) / total if total else None,
        "accepted_correct": accepted_correct,
        "accepted_incorrect": accepted_incorrect,
        "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
        "originally_correct": original_correct,
        "originally_correct_rejected": correct_rejected,
        "originally_correct_rejected_percent": (
            100.0 * correct_rejected / original_correct if original_correct else None
        ),
    }


def evaluate_split(*, split: str, model_path: Path, local_root: Path, food101_root: Path,
                   threshold: float | None = None) -> dict:
    _, datasets, metadata = create_tf_phase5_datasets(
        local_root=local_root, food101_root=food101_root, batch_size=32, seed=42,
    )
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    dataset = datasets[split]
    expected_size = 2719 if split == "test" else 1700
    if len(dataset) != expected_size:
        raise ValueError(f"Expected {expected_size} {split} images, got {len(dataset)}")

    classifier = TFLiteFoodClassifier(model_path)
    rows = []
    for index in range(len(dataset)):
        image, true_index = dataset[index]
        output = classifier.predict_array(_model_input_from_torch_image(image), top_k=1)
        predicted_class = output["predicted_class"]
        true_class = PHASE5_CLASSES[int(true_index)]
        confidence = float(output["confidence"])
        rows.append({
            "index": index,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "maximum_softmax_confidence": confidence,
            "correct": predicted_class == true_class,
        })

    correct = [
        {"confidence": row["maximum_softmax_confidence"]}
        for row in rows if row["correct"]
    ]
    incorrect = [
        {"confidence": row["maximum_softmax_confidence"]}
        for row in rows if not row["correct"]
    ]
    report = {
        "model": str(Path(model_path).resolve().relative_to(ROOT)),
        "split": split,
        "class_order": list(PHASE5_CLASSES),
        "preprocessing": (
            "Existing Phase 5 deterministic evaluation transform: RGB, resize shortest edge "
            "to 256 with bilinear interpolation, center crop 224x224, then invert the "
            "verified ImageNet normalization to float32 [0,255] for TFLite."
        ),
        "dataset_counts": metadata["class_counts"][split],
        "sample_count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "confidence_when_correct": confidence_statistics(correct),
        "confidence_when_incorrect": confidence_statistics(incorrect),
        "predictions": rows,
    }
    if split == "validation":
        report["candidate_threshold_metrics"] = [
            threshold_metrics(rows, value) for value in CANDIDATE_THRESHOLDS
        ]
    else:
        if threshold is None:
            raise ValueError("test evaluation requires the validation-selected --threshold")
        report["frozen_threshold_metrics"] = threshold_metrics(rows, threshold)
    return report


def write_report(report: dict, path: Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _self_test() -> None:
    rows = [
        {"confidence": 0.8, "correct": True},
        {"confidence": 0.4, "correct": True},
        {"confidence": 0.5, "correct": False},
        {"confidence": 0.2, "correct": False},
    ]
    assert confidence_statistics(rows[:2])["count"] == 2
    metrics = threshold_metrics(rows, 0.5)
    assert metrics["accepted"] == 2 and metrics["rejected"] == 2
    assert metrics["accepted_correct"] == 1 and metrics["accepted_incorrect"] == 1
    assert metrics["accepted_accuracy"] == 0.5
    assert metrics["originally_correct_rejected"] == 1
    assert metrics["originally_correct_rejected_percent"] == 50.0
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        try:
            threshold_metrics(rows, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid threshold accepted: {invalid}")
    print("Confidence metric self-checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--food101-root", type=Path, default=DATA_DIR)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.split == "test" and args.threshold is None:
        parser.error("--threshold is required for held-out test evaluation")
    report = evaluate_split(
        split=args.split, model_path=args.model, local_root=args.local_root,
        food101_root=args.food101_root, threshold=args.threshold,
    )
    default_report = DEFAULT_VALIDATION_REPORT if args.split == "validation" else DEFAULT_TEST_REPORT
    report_path = args.report or default_report
    write_report(report, report_path)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    print(f"Full per-image report: {Path(report_path).resolve()}")


if __name__ == "__main__":
    main()
