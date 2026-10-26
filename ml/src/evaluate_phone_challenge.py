"""Evaluate one or more 13-class TFLite models on real-phone challenge images.

Expected tree: ROOT/<canonical_class>/<session>/<variation>/<image>.jpg. Challenge
images are evaluated only and are never fed to training or validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .dataset import PHASE5_CLASSES
    from .mobilenet_v3_pipeline import TFLiteFoodClassifier
else:
    from dataset import PHASE5_CLASSES
    from mobilenet_v3_pipeline import TFLiteFoodClassifier


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def challenge_images(root: Path) -> list[tuple[Path, str, str, str]]:
    root = Path(root).expanduser().resolve()
    records = []
    for label in PHASE5_CLASSES:
        label_root = root / label
        if not label_root.is_dir():
            continue
        for path in sorted(path for path in label_root.rglob("*") if path.suffix.lower() in EXTENSIONS):
            parts = path.relative_to(label_root).parts
            session = parts[0] if len(parts) > 1 else "unspecified_session"
            variation = parts[1] if len(parts) > 2 else "unspecified_variation"
            records.append((path, label, session, variation))
    if not records:
        raise ValueError(f"No labeled challenge images found under {root}")
    return records


def evaluate(models: list[Path], root: Path, *, threshold: float = 0.55) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    images = challenge_images(root)
    output = {
        "split": "real_phone_challenge", "training_or_validation_use": False,
        "threshold": threshold, "sample_count": len(images), "variation_counts": {}, "models": {},
    }
    for path, _, _, variation in images:
        output["variation_counts"][variation] = output["variation_counts"].get(variation, 0) + 1
    for model_path in models:
        classifier = TFLiteFoodClassifier(model_path)
        rows = []
        for path, true_class, session, variation in images:
            prediction = classifier.predict(path, top_k=1)
            rows.append({
                "path": str(path), "true_class": true_class,
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "correct": prediction["predicted_class"] == true_class,
                "accepted_at_threshold": prediction["confidence"] >= threshold,
                "session": session, "variation": variation,
            })
        per_class = {}
        for label in PHASE5_CLASSES:
            chosen = [row for row in rows if row["true_class"] == label]
            accepted = [row for row in chosen if row["accepted_at_threshold"]]
            per_class[label] = {
                "samples": len(chosen),
                "top1_accuracy": sum(row["correct"] for row in chosen) / len(chosen) if chosen else None,
                "recall": sum(row["correct"] for row in chosen) / len(chosen) if chosen else None,
                "false_rejections": sum(row["correct"] and not row["accepted_at_threshold"] for row in chosen),
                "false_rejection_rate_of_correct": (
                    sum(row["correct"] and not row["accepted_at_threshold"] for row in chosen)
                    / sum(row["correct"] for row in chosen) if any(row["correct"] for row in chosen) else None
                ),
                "confidence_correct": _summary([row["confidence"] for row in chosen if row["correct"]]),
                "confidence_incorrect": _summary([row["confidence"] for row in chosen if not row["correct"]]),
                "variation_accuracy": {
                    variation: _accuracy([row for row in chosen if row["variation"] == variation])
                    for variation in sorted({row["variation"] for row in chosen})
                },
            }
        output["models"][str(model_path)] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "accepted_rate": sum(row["accepted_at_threshold"] for row in rows) / len(rows),
            "correct_rejected": sum(row["correct"] and not row["accepted_at_threshold"] for row in rows),
            "wrong_accepted": sum((not row["correct"]) and row["accepted_at_threshold"] for row in rows),
            "per_class": per_class, "predictions": rows,
        }
    return output


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def _accuracy(rows: list[dict]) -> float | None:
    return sum(row["correct"] for row in rows) / len(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("ml/data/real_phone_challenge"))
    parser.add_argument("--model", type=Path, action="append", required=True,
                        help="TFLite 13-class model; repeat to compare on identical images")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")
    result = evaluate(args.model, args.root, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": result["sample_count"], "models": {
        model: {"accuracy": metrics["accuracy"], "accepted_rate": metrics["accepted_rate"]}
        for model, metrics in result["models"].items()
    }, "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
