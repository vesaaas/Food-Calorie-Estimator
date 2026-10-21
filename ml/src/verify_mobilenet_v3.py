"""Verify the trained MobileNetV3 checkpoint and real TFLite flatbuffer offline."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

if __package__:
    from .dataset import DATA_DIR, PHASE5_CLASSES
    from .evaluate_mobilenet_v3 import predict_dataset
    from .export_mobilenet_v3_tflite import CONTRACT_PATH, DEFAULT_TFLITE, LABELS_PATH
    from .mobilenet_v3 import IMAGE_SIZE
    from .mobilenet_v3_pipeline import TFLiteFoodClassifier, TFLiteImageNutritionPipeline, preprocess_image
    from .tf_phase5_data import _model_input_from_torch_image, create_tf_phase5_datasets
    from .train_mobilenet_v3 import SELECTED_CHECKPOINT, SELECTED_HISTORY
    from .download_phase5_dataset import DEFAULT_OUTPUT
else:
    from dataset import DATA_DIR, PHASE5_CLASSES
    from evaluate_mobilenet_v3 import predict_dataset
    from export_mobilenet_v3_tflite import CONTRACT_PATH, DEFAULT_TFLITE, LABELS_PATH
    from mobilenet_v3 import IMAGE_SIZE
    from mobilenet_v3_pipeline import TFLiteFoodClassifier, TFLiteImageNutritionPipeline, preprocess_image
    from tf_phase5_data import _model_input_from_torch_image, create_tf_phase5_datasets
    from train_mobilenet_v3 import SELECTED_CHECKPOINT, SELECTED_HISTORY
    from download_phase5_dataset import DEFAULT_OUTPUT

from nutrition.src.catalog import load_catalog
from nutrition.src.calculator import estimate


ROOT = Path(__file__).resolve().parents[2]
PHASE3_ARTIFACTS = (
    ROOT / "ml/artifacts/efficientnet_b0_best.pt",
    ROOT / "ml/artifacts/phase3_final_test.json",
)
PHASE5_ARTIFACTS = (
    ROOT / "ml/artifacts/efficientnet_b0_phase5_best.pt",
    ROOT / "ml/artifacts/phase5_mixed_test.json",
)
DEFAULT_REPORT = ROOT / "ml/artifacts/mobilenet_v3_small_phase7_v2_tflite_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(
    *, checkpoint=SELECTED_CHECKPOINT, model_path=DEFAULT_TFLITE,
    contract_path=CONTRACT_PATH, history_path=SELECTED_HISTORY,
    local_root=DEFAULT_OUTPUT, food101_root=DATA_DIR, representative_per_class=3,
    report_path=DEFAULT_REPORT,
):
    checkpoint, model_path = Path(checkpoint).resolve(), Path(model_path).resolve()
    contract_path, history_path = Path(contract_path).resolve(), Path(history_path).resolve()
    for path in (checkpoint, model_path, contract_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required MobileNetV3 artifact is missing: {path}")
    baseline_paths = tuple(path for path in (*PHASE3_ARTIFACTS, *PHASE5_ARTIFACTS) if path.is_file())
    baseline_before = {str(path): sha256(path) for path in baseline_paths}

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = list(PHASE5_CLASSES)
    if contract.get("labels") != expected or len(set(contract["labels"])) != 13:
        raise ValueError("Mobile metadata must contain each canonical classifier label exactly once in order")
    if Path(LABELS_PATH).read_text(encoding="utf-8").splitlines() != expected:
        raise ValueError("Standalone Flutter labels file differs from the canonical class order")
    if contract.get("model_file") != model_path.relative_to(ROOT).as_posix():
        raise ValueError("Contract model path does not match the selected TFLite model")
    if contract["input"] != {
        "width": IMAGE_SIZE, "height": IMAGE_SIZE, "color": "RGB", "dtype": "float32",
        "tensor_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3], "layout": "NHWC", "pixel_range": [0, 255],
        "resize": "Resize shortest edge to 256 pixels with bilinear interpolation, then center crop 224x224",
        "normalization": "The model includes Keras MobileNetV3 rescaling from [0,255] to [-1,1]; callers must not apply ImageNet mean/std normalization.",
    }:
        raise ValueError("Input preprocessing metadata does not match the verified model input")
    if contract["output"]["interpretation"].find("softmax probabilities") < 0:
        raise ValueError("Contract output interpretation is not softmax probabilities")
    training_metadata = json.loads(history_path.read_text(encoding="utf-8"))
    if training_metadata.get("classes") != expected or training_metadata.get("selection_metric") != "validation_accuracy":
        raise ValueError("Training history class order or model-selection metric is inconsistent")

    model = tf.keras.models.load_model(checkpoint, compile=False)
    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Keras input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, len(PHASE5_CLASSES)):
        raise ValueError(f"Keras output shape mismatch: {model.output_shape}")
    classifier = TFLiteFoodClassifier(model_path, contract_path)
    catalog = load_catalog(expected_classes=PHASE5_CLASSES)

    _, source_datasets, split_metadata = create_tf_phase5_datasets(
        local_root=local_root, food101_root=food101_root, batch_size=32, seed=42,
    )
    test_dataset = source_datasets["test"]
    if len(test_dataset) != 2719:
        raise ValueError(f"Expected the existing 2,719-image Phase 5 mixed test set, got {len(test_dataset)}")
    targets = test_dataset.targets
    selected_indices = []
    for class_index in range(len(PHASE5_CLASSES)):
        class_indices = [i for i, label in enumerate(targets) if label == class_index]
        if not class_indices:
            raise ValueError(f"Mixed test set lacks class {PHASE5_CLASSES[class_index]}")
        selected_indices.extend(class_indices[:representative_per_class])
    arrays, labels = [], []
    for index in selected_indices:
        image, label = test_dataset[index]
        arrays.append(_model_input_from_torch_image(image))
        labels.append(int(label))
    batch = np.stack(arrays).astype(np.float32)
    keras_probabilities = model(batch, training=False).numpy()
    if keras_probabilities.shape != (len(arrays), len(PHASE5_CLASSES)):
        raise ValueError(f"Unexpected Keras output shape: {keras_probabilities.shape}")
    lite_probabilities = []
    lite_decoded = []
    for image in arrays:
        prediction = classifier.predict_array(image, top_k=1)
        lite_probabilities.append(prediction["probabilities"])
        lite_decoded.append(prediction["predicted_class"])
    lite_probabilities = np.asarray(lite_probabilities)
    if not np.isfinite(keras_probabilities).all() or not np.isfinite(lite_probabilities).all():
        raise ValueError("Keras or TFLite produced nonfinite output")
    if not np.allclose(keras_probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Keras outputs are not softmax probabilities")
    max_abs_error = float(np.max(np.abs(keras_probabilities - lite_probabilities)))
    mean_abs_error = float(np.mean(np.abs(keras_probabilities - lite_probabilities)))
    keras_labels = keras_probabilities.argmax(axis=1)
    matching = int(sum(label == expected[class_index] for label, class_index in zip(lite_decoded, keras_labels, strict=True)))
    if max_abs_error > 0.4 or mean_abs_error > 0.03 or matching / len(arrays) < 0.9:
        raise ValueError(
            f"TFLite/Keras comparison exceeds tolerance: top1={matching}/{len(arrays)}, "
            f"max_abs={max_abs_error}, mean_abs={mean_abs_error}"
        )
    if any(label not in PHASE5_CLASSES for label in lite_decoded):
        raise ValueError("TFLite prediction decoded to a noncanonical nutrition ID")

    tflite_correct = 0
    tflite_class_correct = np.zeros(len(PHASE5_CLASSES), dtype=np.int64)
    tflite_class_count = np.zeros(len(PHASE5_CLASSES), dtype=np.int64)
    for index in range(len(test_dataset)):
        image, true_label = test_dataset[index]
        result = classifier.predict_array(_model_input_from_torch_image(image), top_k=1)
        predicted_index = PHASE5_CLASSES.index(result["predicted_class"])
        tflite_class_count[int(true_label)] += 1
        tflite_class_correct[int(true_label)] += int(predicted_index == int(true_label))
        tflite_correct += int(predicted_index == int(true_label))

    # Also exercise real file decoding and the exact published mobile preprocessing.
    first_source = test_dataset.datasets[0]
    sample_image_path = first_source.image_paths[0]
    try:
        preprocess_image(ROOT / "ml/artifacts/this_image_must_not_exist.png")
    except FileNotFoundError:
        pass
    else:
        raise ValueError("Missing image paths must be rejected")
    source_tensor, _ = test_dataset[0]
    mobile_pixels = preprocess_image(sample_image_path)[0]
    training_contract_pixels = _model_input_from_torch_image(source_tensor)
    if not np.allclose(mobile_pixels, training_contract_pixels, atol=1.0):
        raise ValueError("Python mobile preprocessing differs from the verified Phase 5 evaluation transform")
    image_prediction = classifier.predict(sample_image_path, top_k=3)
    pipeline_prediction = TFLiteImageNutritionPipeline(classifier, catalog).predict(sample_image_path)
    if image_prediction["predicted_class"] not in catalog.food_ids:
        raise ValueError("TFLite image prediction did not map to the exact nutrition catalog")
    if pipeline_prediction["food_id"] != image_prediction["predicted_class"]:
        raise ValueError("TFLite image-to-nutrition handoff changed the predicted class")

    # The image-to-nutrition handoff uses only the exact food ID; confidence is never
    # passed to the nutrition calculator and cannot alter its portion or macros.
    class_id = lite_decoded[0]
    nutrition = estimate(class_id, catalog=catalog)
    class_record = catalog.lookup(class_id)
    if nutrition["portion_grams"] <= 0 or class_record["food_id"] != class_id:
        raise ValueError("Canonical class did not map to a valid nutrition serving")

    class FixedClassifier:
        def __init__(self, confidence):
            self.confidence = confidence

        def predict(self, _image_path, *, top_k=3):
            return {"predicted_class": class_id, "confidence": self.confidence,
                    "top_predictions": [{"class": class_id, "confidence": self.confidence}]}

    first = TFLiteImageNutritionPipeline(FixedClassifier(0.01), catalog).predict("ignored.jpg")
    second = TFLiteImageNutritionPipeline(FixedClassifier(0.99), catalog).predict("ignored.jpg")
    if first["portion_grams"] != second["portion_grams"] or first["nutrients"] != second["nutrients"]:
        raise ValueError("Classifier confidence changed portion or nutrition values")

    baseline_after = {str(path): sha256(path) for path in baseline_paths}
    if baseline_before != baseline_after:
        raise RuntimeError("A historical EfficientNet checkpoint or evaluation artifact changed")
    result = {
        "model": str(model_path), "model_size_bytes": model_path.stat().st_size,
        "input_shape": classifier.input_detail["shape"].tolist(),
        "input_dtype": str(classifier.input_detail["dtype"]),
        "output_shape": classifier.output_detail["shape"].tolist(),
        "output_dtype": str(classifier.output_detail["dtype"]),
        "classes": expected, "mixed_test_images": len(test_dataset),
        "representative_test_samples": len(arrays), "keras_tflite_top1_matches": matching,
        "max_abs_output_difference": max_abs_error, "mean_abs_output_difference": mean_abs_error,
        "tflite_test_accuracy": tflite_correct / len(test_dataset),
        "tflite_test_per_class": {
            label: {"correct": int(tflite_class_correct[index]),
                    "samples": int(tflite_class_count[index]),
                    "accuracy": float(tflite_class_correct[index] / tflite_class_count[index])}
            for index, label in enumerate(PHASE5_CLASSES)
        },
        "representative_per_class": representative_per_class,
        "split_counts": split_metadata["class_counts"],
        "nutrition_lookup_food_id": class_id,
        "historical_artifacts_unchanged": baseline_before == baseline_after,
        "historical_artifact_sha256": baseline_after,
    }
    report_path = Path(report_path).expanduser()
    result["verification_report"] = str(report_path.resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=report_path.parent,
        prefix=f".{report_path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    try:
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--history", type=Path, default=SELECTED_HISTORY)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--food101-root", type=Path, default=DATA_DIR)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        result = verify(
            checkpoint=args.checkpoint, model_path=args.model, contract_path=args.contract,
            history_path=args.history, local_root=args.local_root,
            food101_root=args.food101_root, representative_per_class=args.samples_per_class,
            report_path=args.report,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"MobileNetV3/TFLite verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
