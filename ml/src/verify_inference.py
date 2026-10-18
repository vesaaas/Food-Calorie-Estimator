"""Verify single-image prediction on temporary synthetic PIL images only."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

if __package__:
    from . import predict as prediction_cli
    from .checkpoint import DEFAULT_CHECKPOINT
    from .inference import FoodImagePredictor
else:
    import predict as prediction_cli
    from checkpoint import DEFAULT_CHECKPOINT
    from inference import FoodImagePredictor


class InferenceChecks(unittest.TestCase):
    checkpoint_path = DEFAULT_CHECKPOINT
    requested_device = "cpu"

    def test_checkpoint_single_image_probabilities_mapping_and_immutability(self):
        def checkpoint_digest():
            with self.checkpoint_path.open("rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()

        original_digest = checkpoint_digest()
        predictor = FoodImagePredictor.from_checkpoint(
            self.checkpoint_path, requested_device=self.requested_device,
        )
        self.assertFalse(predictor.model.training)
        self.assertEqual(predictor.image_size, 224)
        self.assertEqual(len(predictor.class_to_idx), 10)
        self.assertEqual(predictor.index_to_class, {index: name for name, index in predictor.class_to_idx.items()})
        parameters_before = {name: value.detach().cpu().clone() for name, value in predictor.model.state_dict().items()}
        calls = []

        def verify_forward(module, inputs, output):
            self.assertFalse(module.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(output.requires_grad)
            self.assertEqual(tuple(inputs[0].shape), (1, 3, 224, 224))
            calls.append(True)

        hook = predictor.model.register_forward_hook(verify_forward)
        try:
            with tempfile.TemporaryDirectory(prefix="food-inference-") as directory:
                rgba_path = Path(directory) / "synthetic.png"
                Image.new("RGBA", (97, 61), color=(160, 90, 40, 100)).save(rgba_path)
                image = predictor.preprocess_image(rgba_path)
                self.assertEqual(tuple(image.shape), (3, 224, 224))
                self.assertEqual(image.dtype, torch.float32)
                self.assertTrue(torch.isfinite(image).all().item())
                prediction = predictor.predict(rgba_path, top_k=3)
                grayscale_path = Path(directory) / "synthetic-gray.png"
                Image.new("L", (97, 61), color=127).save(grayscale_path)
                gray_prediction = predictor.predict(grayscale_path, top_k=10)
                self.assertEqual(len(gray_prediction["top_predictions"]), 10)
                gray_probabilities = [item["probability"] for item in gray_prediction["top_predictions"]]
                self.assertTrue(all(left >= right for left, right in zip(gray_probabilities, gray_probabilities[1:])))
                self.assertEqual(gray_prediction["class_index"], predictor.class_to_idx[gray_prediction["predicted_class"]])
                all_predictions = predictor.predict(rgba_path, top_k=10)
                output = io.StringIO()
                with (
                    patch.object(FoodImagePredictor, "from_checkpoint", return_value=predictor) as load,
                    patch("sys.argv", [
                        "predict.py", str(rgba_path), "--checkpoint", str(self.checkpoint_path),
                        "--device", self.requested_device, "--top-k", "3",
                    ]),
                    redirect_stdout(output),
                ):
                    prediction_cli.main()
                load.assert_called_once()
                self.assertIn(f"Prediction: {prediction['predicted_class']}", output.getvalue())
                self.assertIn("Top predictions:", output.getvalue())
                with self.assertRaises(ValueError):
                    predictor.predict(rgba_path, top_k=11)
                with self.assertRaises(ValueError):
                    predictor.predict(rgba_path, top_k=0)
                invalid_path = Path(directory) / "not-an-image.png"
                invalid_path.write_text("not an image")
                with self.assertRaises(ValueError):
                    predictor.predict(invalid_path)
                with self.assertRaises(FileNotFoundError):
                    predictor.predict(Path(directory) / "missing.png")
        finally:
            hook.remove()
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(prediction["top_predictions"]), 3)
        probabilities = [item["probability"] for item in prediction["top_predictions"]]
        self.assertTrue(all(torch.isfinite(torch.tensor(value)).item() for value in probabilities))
        self.assertTrue(all(left >= right for left, right in zip(probabilities, probabilities[1:])))
        all_probabilities = [item["probability"] for item in all_predictions["top_predictions"]]
        self.assertAlmostEqual(sum(all_probabilities), 1.0, places=6)
        self.assertTrue(all(left >= right for left, right in zip(all_probabilities, all_probabilities[1:])))
        self.assertEqual(prediction["class_index"], predictor.class_to_idx[prediction["predicted_class"]])
        self.assertTrue(all(parameter.grad is None for parameter in predictor.model.parameters()))
        self.assertTrue(all(torch.equal(value, predictor.model.state_dict()[name].cpu()) for name, value in parameters_before.items()))
        self.assertEqual(checkpoint_digest(), original_digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    arguments = parser.parse_args()
    InferenceChecks.checkpoint_path = arguments.checkpoint.expanduser()
    InferenceChecks.requested_device = arguments.device
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(InferenceChecks),
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("PASS: synthetic single-image inference; no dataset images or metrics; no optimizer step; checkpoint unchanged")


if __name__ == "__main__":
    main()
