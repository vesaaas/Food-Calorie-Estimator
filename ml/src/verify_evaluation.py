"""Verify evaluation with synthetic data only; no Food-101 access or training."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__:
    from . import dataset as dataset_module, evaluate as evaluation
    from .checkpoint import DEFAULT_CHECKPOINT, load_checkpoint, preprocessing_metadata
    from .dataset import DataConfig, DEVELOPMENT_CLASSES, create_dataloaders
    from .preprocessing import build_transforms
    from .runtime import select_device
else:
    import dataset as dataset_module
    import evaluate as evaluation
    from checkpoint import DEFAULT_CHECKPOINT, load_checkpoint, preprocessing_metadata
    from dataset import DataConfig, DEVELOPMENT_CLASSES, create_dataloaders
    from preprocessing import build_transforms
    from runtime import select_device


class SyntheticTestDataset(Dataset):
    """In-memory protocol fixture; all paths are invented and never opened."""

    def __init__(self) -> None:
        self.classes = DEVELOPMENT_CLASSES
        self.class_to_idx = {name: label for label, name in enumerate(self.classes)}
        self.targets = tuple(label for label in range(len(self.classes)) for _ in range(250))
        self.indices = tuple(range(len(self.targets)))
        self.image_paths = tuple(Path("/synthetic-food") / self.classes[label] / f"{index}.jpg" for index, label in enumerate(self.targets))
        self.source = SimpleNamespace(
            _split="test", classes=list(self.classes),
            _image_files=list(self.image_paths), _labels=list(self.targets),
        )
        self.transform = build_transforms()[1]
        self.visited = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.visited.append(index)
        logits = torch.zeros(len(self.classes))
        logits[self.targets[index]] = 2
        return logits, self.targets[index]


def synthetic_metadata() -> dict:
    config = DataConfig()
    data_config = asdict(config)
    data_config.update(root=str(config.root), classes=list(config.classes))
    return {
        "classes": list(config.classes),
        "class_to_idx": {name: label for label, name in enumerate(config.classes)},
        "data_config": data_config, "preprocessing": preprocessing_metadata(config.image_size),
        "training": {"epoch": 4, "validation_metrics": {"accuracy": 0.92}},
        "model_name": "synthetic_fixture", "model_config": {"num_classes": len(config.classes)},
    }


class EvaluationChecks(unittest.TestCase):
    checkpoint_path = DEFAULT_CHECKPOINT
    requested_device = "cpu"

    def test_metrics_and_matrix_orientation_with_uneven_batches(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 1])
        predictions = [0, 2, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 1]
        logits = torch.zeros(len(labels), len(DEVELOPMENT_CLASSES))
        logits[torch.arange(len(labels)), predictions] = 2
        batches = [(logits[start:start + 5], labels[start:start + 5]) for start in range(0, len(labels), 5)]
        result = evaluation.evaluate_model(nn.Identity(), batches, torch.device("cpu"), DEVELOPMENT_CLASSES)
        self.assertEqual(result["samples"], 13)
        self.assertEqual(result["correct"], 10)
        self.assertEqual(result["accuracy"], 10 / 13)
        self.assertAlmostEqual(result["loss"], math.log(math.exp(2) + 9) - 2 * 10 / 13, places=6)
        expected = [[0] * len(DEVELOPMENT_CLASSES) for _ in DEVELOPMENT_CLASSES]
        for label, predicted in zip(labels.tolist(), predictions, strict=True):
            expected[label][predicted] += 1
        self.assertEqual(result["confusion_matrix"], expected)
        self.assertEqual(result["confusion_matrix_axes"], {"rows": "true", "columns": "predicted"})
        self.assertEqual(list(result["per_class"]), list(DEVELOPMENT_CLASSES))
        for label, name in enumerate(DEVELOPMENT_CLASSES):
            count = labels.tolist().count(label)
            self.assertEqual(result["per_class"][name]["samples"], count)
            self.assertEqual(result["per_class"][name]["accuracy"], expected[label][label] / count)

    def test_empty_missing_classes_and_invalid_batches(self):
        device = torch.device("cpu")
        with self.assertRaises(ValueError):
            evaluation.evaluate_model(nn.Identity(), [], device, DEVELOPMENT_CLASSES)
        result = evaluation.evaluate_model(nn.Identity(), [(torch.zeros(1, 10), torch.tensor([0]))], device, DEVELOPMENT_CLASSES)
        self.assertIsNone(result["per_class"][DEVELOPMENT_CLASSES[1]]["accuracy"])
        for logits, labels, error in (
            (torch.zeros(1, 9), torch.tensor([0]), ValueError),
            (torch.zeros(1, 10), torch.tensor([10]), ValueError),
            (torch.zeros(1, 10), torch.tensor([-1]), ValueError),
            (torch.zeros(1, 10), torch.tensor([0.0]), ValueError),
            (torch.full((1, 10), float("nan")), torch.tensor([0]), RuntimeError),
        ):
            with self.subTest(error=error, labels=labels), self.assertRaises(error):
                evaluation.evaluate_model(nn.Identity(), [(logits, labels)], device, DEVELOPMENT_CLASSES)

    def test_loader_preflight_rejects_wrong_metadata_or_coverage(self):
        dataset = SyntheticTestDataset()
        loader = create_dataloaders({"test": dataset})["test"]
        metadata = synthetic_metadata()
        evaluation.validate_test_loader(dataset, loader, metadata)
        for attribute, value in (
            ("classes", tuple(reversed(DEVELOPMENT_CLASSES))),
            ("class_to_idx", {}), ("transform", build_transforms()[0]),
            ("indices", dataset.indices[:-1] + (0,)),
            ("image_paths", dataset.image_paths[:-1] + (dataset.image_paths[0],)),
            ("targets", tuple(reversed(dataset.targets))),
        ):
            with self.subTest(attribute=attribute), patch.object(dataset, attribute, value), self.assertRaises(ValueError):
                evaluation.validate_test_loader(dataset, loader, metadata)
        with patch.object(dataset.source, "_split", "train"), self.assertRaises(ValueError):
            evaluation.validate_test_loader(dataset, loader, metadata)
        for wrong_loader in (DataLoader(dataset, batch_size=16, shuffle=True), DataLoader(dataset, batch_size=16, drop_last=True)):
            with self.assertRaises(ValueError):
                evaluation.validate_test_loader(dataset, wrong_loader, metadata)
        self.assertEqual(dataset.visited, [])

    def test_orchestration_visits_all_synthetic_samples_once_and_writes_report(self):
        dataset = SyntheticTestDataset()
        config = DataConfig(batch_size=64)
        loaders = create_dataloaders({"test": dataset}, config)
        metadata = synthetic_metadata()
        with tempfile.TemporaryDirectory(prefix="food-evaluation-") as directory:
            checkpoint = Path(directory) / "synthetic.pt"
            checkpoint.write_bytes(b"synthetic checkpoint marker; loading is mocked")
            before = evaluation.file_sha256(checkpoint)
            destination = Path(directory) / "result.json"
            output = io.StringIO()
            with (
                patch.object(evaluation, "create_data_pipeline", return_value=({"test": dataset}, loaders)) as pipeline,
                patch.object(evaluation, "load_checkpoint", return_value=(nn.Identity(), metadata)) as load,
                redirect_stdout(output),
            ):
                result = evaluation.run_evaluation(checkpoint, requested_device="cpu", batch_size=64, report_path=destination)
            load.assert_called_once_with(checkpoint.resolve(), device=torch.device("cpu"))
            self.assertEqual(pipeline.call_args.args[0], config)
            self.assertEqual(dataset.visited, list(range(2500)))
            self.assertEqual(result["samples"], 2500)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertTrue(all(item["samples"] == 250 for item in result["per_class"].values()))
            self.assertEqual(result["checkpoint_epoch"], 4)
            self.assertEqual(result["checkpoint_sha256"], before)
            self.assertEqual(evaluation.file_sha256(checkpoint), before)
            self.assertEqual(json.loads(destination.read_text()), result)
            self.assertIn("rows=true, columns=predicted", output.getvalue())
            self.assertIn("samples: 2,500", output.getvalue())

    def test_output_paths_cannot_overwrite_checkpoint_or_existing_files(self):
        with tempfile.TemporaryDirectory(prefix="food-evaluation-paths-") as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"must remain unchanged")
            symlink = Path(directory) / "symlink.json"
            symlink.symlink_to(checkpoint)
            hardlink = Path(directory) / "hardlink.json"
            os.link(checkpoint, hardlink)
            existing = Path(directory) / "existing.json"
            existing.write_text("existing report")
            for destination in (checkpoint, symlink, hardlink, existing, Path(directory) / "bad.pt"):
                with self.subTest(destination=destination), patch.object(evaluation, "load_checkpoint") as load:
                    with self.assertRaises((ValueError, FileExistsError)):
                        evaluation.run_evaluation(checkpoint, requested_device="cpu", report_path=destination)
                    load.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"must remain unchanged")
            self.assertEqual(existing.read_text(), "existing report")

    def test_saved_checkpoint_on_synthetic_images_preserves_state_and_gradients(self):
        device = select_device(self.requested_device)
        before = evaluation.file_sha256(self.checkpoint_path)
        model, metadata = load_checkpoint(self.checkpoint_path, device=device)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        first_parameter = next(model.parameters())
        first_parameter.grad = torch.ones_like(first_parameter)
        previous_gradient = first_parameter.grad.clone()
        forward_calls = []

        def check_mode(module, inputs, output):
            self.assertFalse(module.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(output.requires_grad)
            forward_calls.append(True)

        model.train()  # The evaluator must switch BatchNorm/dropout back to eval.
        hook = model.register_forward_hook(check_mode)
        try:
            result = evaluation.evaluate_model(
                model, [(torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]))],
                device, tuple(metadata["classes"]),
            )
        finally:
            hook.remove()
        self.assertEqual(len(forward_calls), 1)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(len(result["confusion_matrix"]), 10)
        self.assertTrue(math.isfinite(result["loss"]))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), state[name]), name)
        self.assertTrue(torch.equal(first_parameter.grad, previous_gradient))
        self.assertTrue(all(parameter.grad is None for parameter in list(model.parameters())[1:]))
        self.assertEqual(evaluation.file_sha256(self.checkpoint_path), before)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    arguments = parser.parse_args()
    EvaluationChecks.checkpoint_path = arguments.checkpoint.expanduser()
    EvaluationChecks.requested_device = arguments.device
    torch.set_num_threads(1)
    # Any accidental real dataset construction fails, even if a test's mock is removed.
    with patch.object(dataset_module, "load_food101", side_effect=AssertionError("Verification must not access Food-101")):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("PASS: synthetic evaluation checks; official images consumed: 0; training steps: 0; checkpoint unchanged")


if __name__ == "__main__":
    main()
