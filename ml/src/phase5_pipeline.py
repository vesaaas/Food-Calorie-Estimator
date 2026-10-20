"""Reusable Phase 5 image classification to default-serving nutrition estimate."""

from pathlib import Path

if __package__:
    from .checkpoint import ARTIFACT_DIR, load_checkpoint
    from .dataset import PHASE5_CLASSES
    from .inference import FoodImagePredictor
    from .runtime import select_device
else:
    from checkpoint import ARTIFACT_DIR, load_checkpoint
    from dataset import PHASE5_CLASSES
    from inference import FoodImagePredictor
    from runtime import select_device

from nutrition.src.calculator import estimate
from nutrition.src.catalog import load_catalog

DEFAULT_PHASE5_CHECKPOINT = ARTIFACT_DIR / "efficientnet_b0_phase5_best.pt"


class ImageNutritionPipeline:
    """Classifier confidence is returned separately and never affects nutrition."""

    def __init__(self, predictor, catalog):
        self.predictor = predictor
        self.catalog = catalog

    @classmethod
    def from_checkpoint(cls, checkpoint=DEFAULT_PHASE5_CHECKPOINT, *, requested_device="auto"):
        device = select_device(requested_device)
        model, metadata = load_checkpoint(checkpoint, device=device, expected_classes=PHASE5_CLASSES)
        predictor = FoodImagePredictor(model, class_to_idx=metadata["class_to_idx"],
                                       image_size=metadata["preprocessing"]["image_size"], device=device)
        return cls(predictor, load_catalog(expected_classes=PHASE5_CLASSES))

    def predict(self, image_path: Path, *, top_k=1) -> dict:
        prediction = self.predictor.predict(image_path, top_k=top_k)
        food_id = prediction["predicted_class"]
        nutrition = estimate(food_id, catalog=self.catalog)
        return {
            "food_id": food_id, "display_name": nutrition["display_name"],
            "confidence": prediction["confidence"], "top_predictions": prediction["top_predictions"],
            "portion_grams": nutrition["portion_grams"], "portion_mode": "default",
            "nutrients": nutrition["nutrients"], "nutrition_source": nutrition["nutrition_source"],
            "catalog_version": nutrition["catalog_version"],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PHASE5_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    try:
        result = ImageNutritionPipeline.from_checkpoint(args.checkpoint, requested_device=args.device).predict(args.image)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Image-to-nutrition prediction failed: {error}\n")
    nutrients = result["nutrients"]
    print(f"Predicted food: {result['display_name']} ({result['food_id']})")
    print(f"Classifier confidence: {result['confidence']:.1%}")
    print(f"Portion: {result['portion_grams']:.1f} g (configured default serving)")
    print(f"Calories: {nutrients['energy_kcal']:.1f} kcal")
    print(f"Protein: {nutrients['protein_g']:.1f} g")
    print(f"Carbohydrates: {nutrients['carbohydrates_g']:.1f} g")
    print(f"Fat: {nutrients['fat_g']:.1f} g")


if __name__ == "__main__":
    main()
