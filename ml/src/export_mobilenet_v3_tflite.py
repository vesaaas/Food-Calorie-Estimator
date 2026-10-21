"""Export the selected Keras MobileNetV3 checkpoint to a real TFLite flatbuffer."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
    from .mobilenet_v3 import IMAGE_SIZE, MODEL_NAME
    from .train_mobilenet_v3 import SELECTED_CHECKPOINT
else:
    from dataset import PHASE5_CLASSES
    from mobilenet_v3 import IMAGE_SIZE, MODEL_NAME
    from train_mobilenet_v3 import SELECTED_CHECKPOINT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFLITE = Path(__file__).resolve().parents[1] / "artifacts" / "mobilenet_v3_small_phase7_v2.tflite"
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "mobile" / "mobilenet_v3_tflite_contract_v2.json"
LABELS_PATH = Path(__file__).resolve().parents[1] / "mobile" / "mobilenet_v3_labels.txt"


def contract_for(model_path: Path, *, output_dtype="float32") -> dict:
    return {
        "contract_version": 1,
        "model_file": Path(model_path).expanduser().resolve().relative_to(ROOT).as_posix(),
        "model_type": "TensorFlow Lite MobileNetV3 Small, 13-class image classifier",
        "architecture": MODEL_NAME,
        "input": {
            "width": IMAGE_SIZE, "height": IMAGE_SIZE, "color": "RGB",
            "dtype": "float32", "tensor_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3],
            "layout": "NHWC", "pixel_range": [0, 255],
            "resize": "Resize shortest edge to 256 pixels with bilinear interpolation, then center crop 224x224",
            "normalization": "The model includes Keras MobileNetV3 rescaling from [0,255] to [-1,1]; callers must not apply ImageNet mean/std normalization.",
        },
        "output": {
            "tensor_shape": [1, len(PHASE5_CLASSES)], "dtype": output_dtype,
            "interpretation": "13 softmax probabilities in labels order; outputs sum approximately to 1",
            "confidence": "Top softmax value is classifier confidence only; it is not a calibrated probability, nutrition factor, or portion estimate.",
        },
        "labels_file": Path(LABELS_PATH).resolve().relative_to(ROOT).as_posix(),
        "labels": list(PHASE5_CLASSES),
        "nutrition_food_id": "exactly the corresponding label string",
    }


def export(checkpoint=SELECTED_CHECKPOINT, output=DEFAULT_TFLITE, *, contract_path=CONTRACT_PATH, labels_path=LABELS_PATH):
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    contract_path, labels_path = Path(contract_path), Path(labels_path)
    for path in (output, contract_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing deployment artifact: {path}")
    if labels_path.exists() and labels_path.read_text(encoding="utf-8").splitlines() != list(PHASE5_CLASSES):
        raise FileExistsError(f"Refusing to overwrite a labels file with different class contents: {labels_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best MobileNetV3 checkpoint does not exist: {checkpoint}")
    model = tf.keras.models.load_model(checkpoint, compile=False)
    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected Keras input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, len(PHASE5_CLASSES)):
        raise ValueError(f"Unexpected Keras output shape: {model.output_shape}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    flatbuffer = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
        temporary_model = Path(handle.name)
        handle.write(flatbuffer)
        handle.flush()
    try:
        # Validate the actual serialized flatbuffer before publishing it.
        interpreter = tf.lite.Interpreter(model_path=str(temporary_model))
        interpreter.allocate_tensors()
        input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if tuple(input_detail["shape"]) != (1, IMAGE_SIZE, IMAGE_SIZE, 3) or input_detail["dtype"] != np.float32:
            raise ValueError(f"TFLite input contract mismatch: {input_detail}")
        if tuple(output_detail["shape"]) != (1, len(PHASE5_CLASSES),) or output_detail["dtype"] != np.float32:
            raise ValueError(f"TFLite output contract mismatch: {output_detail}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_model.replace(output)
    finally:
        temporary_model.unlink(missing_ok=True)
    contract = contract_for(output)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if not labels_path.exists():
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text("\n".join(PHASE5_CLASSES) + "\n", encoding="utf-8")
    result = {"model": str(output), "size_bytes": output.stat().st_size,
              "contract": str(contract_path), "labels": str(labels_path)}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output, contract_path=args.contract, labels_path=args.labels)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"TFLite export failed: {error}\n")


if __name__ == "__main__":
    main()
