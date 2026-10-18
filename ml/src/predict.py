"""Predict the food class in a single image using the saved best checkpoint."""

import argparse
from pathlib import Path

if __package__:
    from .checkpoint import DEFAULT_CHECKPOINT
    from .inference import FoodImagePredictor
else:
    from checkpoint import DEFAULT_CHECKPOINT
    from inference import FoodImagePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one readable PIL-compatible image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    arguments = parser.parse_args()
    try:
        predictor = FoodImagePredictor.from_checkpoint(arguments.checkpoint, requested_device=arguments.device)
        prediction = predictor.predict(arguments.image, top_k=arguments.top_k)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(f"Prediction: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    if arguments.top_k > 1:
        print("Top predictions:")
        for rank, item in enumerate(prediction["top_predictions"], start=1):
            print(f"{rank}. {item['class_name']:24s} {item['probability']:.2%}")


if __name__ == "__main__":
    main()
