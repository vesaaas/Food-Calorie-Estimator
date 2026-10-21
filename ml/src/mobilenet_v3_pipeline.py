"""TFLite MobileNetV3 prediction and exact nutrition handoff."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
    from .export_mobilenet_v3_tflite import DEFAULT_TFLITE, CONTRACT_PATH
else:
    from dataset import PHASE5_CLASSES
    from export_mobilenet_v3_tflite import DEFAULT_TFLITE, CONTRACT_PATH

from nutrition.src.calculator import estimate
from nutrition.src.catalog import load_catalog


def preprocess_image(image_path: Path) -> np.ndarray:
    """Return one NHWC raw RGB float32 image using the published mobile resize rule."""
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if width < 1 or height < 1:
                raise ValueError("Image has empty dimensions")
            scale = 256 / min(width, height)
            resized = (int(width * scale), int(height * scale))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            left = int(round((image.width - 224) / 2.0))
            top = int(round((image.height - 224) / 2.0))
            image = image.crop((left, top, left + 224, top + 224))
            array = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not decode image '{image_path}': {error}") from error
    if array.shape != (224, 224, 3):
        raise ValueError(f"Preprocessor produced unexpected shape {array.shape}")
    return array[None, ...]


class TFLiteFoodClassifier:
    def __init__(self, model_path=DEFAULT_TFLITE, contract_path=CONTRACT_PATH):
        self.model_path = Path(model_path).expanduser().resolve()
        self.contract_path = Path(contract_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"TFLite model does not exist: {self.model_path}")
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        if contract.get("labels") != list(PHASE5_CLASSES):
            raise ValueError("TFLite contract labels differ from the canonical Phase 5 class order")
        self.labels = tuple(contract["labels"])
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        if tuple(self.input_detail["shape"]) != (1, 224, 224, 3) or self.input_detail["dtype"] != np.float32:
            raise ValueError("Unexpected TFLite input tensor contract")
        if tuple(self.output_detail["shape"]) != (1, len(self.labels)) or self.output_detail["dtype"] != np.float32:
            raise ValueError("Unexpected TFLite output tensor contract")

    def predict_array(self, image: np.ndarray, *, top_k: int = 3) -> dict:
        if type(top_k) is not int or not 1 <= top_k <= len(self.labels):
            raise ValueError("top_k must be between 1 and the number of labels")
        image = np.asarray(image, dtype=np.float32)
        if image.shape == (224, 224, 3):
            image = image[None, ...]
        if image.shape != (1, 224, 224, 3) or not np.isfinite(image).all():
            raise ValueError("Input must be one finite [1,224,224,3] RGB float32 image")
        if image.min() < 0 or image.max() > 255:
            raise ValueError("Input RGB values must be in [0,255]")
        self.interpreter.set_tensor(self.input_detail["index"], image)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(self.output_detail["index"])[0]
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
            raise ValueError("TFLite output is not a finite probability vector")
        order = np.argsort(probabilities)[::-1][:top_k]
        return {
            "predicted_class": self.labels[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "top_predictions": [
                {"class": self.labels[int(index)], "confidence": float(probabilities[index])}
                for index in order
            ],
            "probabilities": probabilities,
        }

    def predict(self, image_path: Path, *, top_k: int = 3) -> dict:
        return self.predict_array(preprocess_image(image_path), top_k=top_k)


class TFLiteImageNutritionPipeline:
    """Classifier confidence is informational; portions use nutrition defaults."""

    def __init__(self, classifier: TFLiteFoodClassifier, catalog=None):
        self.classifier = classifier
        self.catalog = catalog or load_catalog(expected_classes=PHASE5_CLASSES)

    def predict(self, image_path: Path, *, top_k=3) -> dict:
        prediction = self.classifier.predict(image_path, top_k=top_k)
        result = estimate(prediction["predicted_class"], catalog=self.catalog)
        return {
            "food_id": prediction["predicted_class"], "display_name": result["display_name"],
            "confidence": prediction["confidence"], "top_predictions": prediction["top_predictions"],
            "portion_grams": result["portion_grams"], "portion_mode": "default",
            "nutrients": result["nutrients"], "nutrition_source": result["nutrition_source"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        pipeline = TFLiteImageNutritionPipeline(TFLiteFoodClassifier(args.model, args.contract))
        result = pipeline.predict(args.image, top_k=args.top_k)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"TFLite image-to-nutrition prediction failed: {error}\n")
    print(f"Predicted food: {result['display_name']} ({result['food_id']})")
    print(f"Classifier confidence: {result['confidence']:.1%} (informational)")
    print(f"Portion: {result['portion_grams']:.1f} g (default serving)")
    for label, key in (("Calories", "energy_kcal"), ("Protein", "protein_g"),
                       ("Carbohydrates", "carbohydrates_g"), ("Fat", "fat_g")):
        print(f"{label}: {result['nutrients'][key]:.1f} {'kcal' if key == 'energy_kcal' else 'g'}")


if __name__ == "__main__":
    main()
