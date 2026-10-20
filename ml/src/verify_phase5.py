"""Offline Phase 5 checks using temporary images and checkpoints only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

if __package__:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from .model import ModelConfig, create_model
    from .phase5_data import validate_manifest
    from .inference import FoodImagePredictor
    from .phase5_pipeline import ImageNutritionPipeline
    from .train_phase5 import initialize_from_phase3
else:
    from checkpoint import load_checkpoint, save_checkpoint
    from dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig
    from model import ModelConfig, create_model
    from phase5_data import validate_manifest
    from inference import FoodImagePredictor
    from phase5_pipeline import ImageNutritionPipeline
    from train_phase5 import initialize_from_phase3

from nutrition.src.catalog import load_catalog
from nutrition.src.calculator import estimate

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class FixedLogits(nn.Module):
    def forward(self, images):
        logits = images.new_zeros((images.shape[0], len(PHASE5_CLASSES)))
        logits[:, 10] = 5.0
        return logits


class Phase5Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(expected_classes=PHASE5_CLASSES)

    def test_exact_order_and_explicit_source_mapping(self):
        self.assertEqual(PHASE5_CLASSES[:10], DEVELOPMENT_CLASSES)
        self.assertEqual(PHASE5_CLASSES[10:], ("borek", "kebap", "sarma"))
        from ml.src.download_phase5_dataset import SOURCE_TO_CANONICAL
        self.assertEqual(SOURCE_TO_CANONICAL, {"borek": "borek", "kebap": "kebap", "yaprak_sarma": "sarma"})

    def test_region_calculations_and_legacy_records(self):
        for label in PHASE5_CLASSES:
            record = self.catalog.lookup(label)
            result = estimate(label, grams=100, catalog=self.catalog)
            self.assertEqual(result["nutrients"], record["per_100g"])
        for label in ("borek", "kebap", "sarma"):
            self.assertGreater(estimate(label, catalog=self.catalog)["portion_grams"], 0)

    def _synthetic_manifest(self, root):
        records = []
        for split_index, split in enumerate(("train", "eval", "test")):
            for label_index, (source_label, canonical) in enumerate((("borek", "borek"), ("kebap", "kebap"), ("yaprak_sarma", "sarma"))):
                path = Path(split) / canonical / "image.jpg"
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                color = (split_index * 50 + label_index * 10, label_index * 50, 10)
                image = Image.new("RGB", (12, 12), color)
                image.save(destination)
                with Image.open(destination) as check:
                    pixel_hash = hashlib.sha256(check.convert("RGB").tobytes()).hexdigest()
                records.append({"path": path.as_posix(), "source_split": split,
                                "source_label": source_label, "canonical_label": canonical,
                                "source_row": split_index * 3 + label_index,
                                "pixel_sha256": pixel_hash, "file_sha256": digest(destination)})
        manifest = {"schema_version": 1, "revision": "3e5026a17d14df81a25f0b6cdb16a58301fbb83c",
                    "source_to_canonical": {"borek": "borek", "kebap": "kebap", "yaprak_sarma": "sarma"},
                    "exported_counts": {split: {"borek": 1, "kebap": 1, "sarma": 1} for split in ("train", "eval", "test")},
                    "records": records}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_manifest_split_isolation_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._synthetic_manifest(root)
            self.assertEqual(len(validate_manifest(root)["records"]), 9)
            manifest["records"][1]["canonical_label"] = "sarma"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_manifest(root)

            manifest = self._synthetic_manifest(root)
            validation_record, test_record = manifest["records"][3], manifest["records"][6]
            validation_path, test_path = root / validation_record["path"], root / test_record["path"]
            test_path.write_bytes(validation_path.read_bytes())
            with Image.open(test_path) as image:
                test_record["pixel_sha256"] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            test_record["file_sha256"] = digest(test_path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leaks between local-food"):
                validate_manifest(root)

    def test_checkpoint_metadata_and_phase3_files_untouched(self):
        phase3_checkpoint = ROOT / "ml/artifacts/efficientnet_b0_best.pt"
        phase3_result = ROOT / "ml/artifacts/phase3_final_test.json"
        if phase3_checkpoint.exists() and phase3_result.exists():
            before = (digest(phase3_checkpoint), digest(phase3_result))
        else:
            before = None
        model = create_model(ModelConfig(num_classes=13, pretrained=False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = DataConfig(root=DATA_DIR, classes=PHASE5_CLASSES)
        training = {"epoch": 1, "config": {"epochs": 1},
                    "train_metrics": {"loss": 2.0, "accuracy": 0.1, "samples": 13},
                    "validation_metrics": {"loss": 2.0, "accuracy": 0.1, "samples": 13},
                    "selection_metric": "validation_accuracy"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase5.pt"
            save_checkpoint(path, model=model, optimizer=optimizer, model_config=ModelConfig(13, False),
                            data_config=config, class_to_idx={name: i for i, name in enumerate(PHASE5_CLASSES)},
                            training=training)
            restored, metadata = load_checkpoint(path, expected_classes=PHASE5_CLASSES)
            self.assertEqual(metadata["classes"], list(PHASE5_CLASSES))
            self.assertEqual(metadata["class_to_idx"]["sarma"], 12)
            self.assertEqual(restored.classifier[1].out_features, 13)
        if before:
            self.assertEqual(before, (digest(phase3_checkpoint), digest(phase3_result)))

    def test_end_to_end_and_confidence_is_informational(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meal.png"
            Image.new("RGB", (256, 256), (120, 30, 20)).save(path)
            predictor = FoodImagePredictor(FixedLogits(), class_to_idx={name: i for i, name in enumerate(PHASE5_CLASSES)},
                                           image_size=224, device=torch.device("cpu"))
            pipeline = ImageNutritionPipeline(predictor, self.catalog)
            result = pipeline.predict(path)
            self.assertEqual(result["food_id"], "borek")
            self.assertEqual(result["portion_grams"], self.catalog.lookup("borek")["default_serving"]["grams"])
            self.assertEqual(result["nutrients"], estimate("borek", catalog=self.catalog)["nutrients"])
            with self.assertRaises(FileNotFoundError):
                pipeline.predict(Path(directory) / "missing.jpg")
            corrupt = Path(directory) / "not-an-image.png"
            corrupt.write_text("not a decodable image")
            with self.assertRaises(ValueError):
                pipeline.predict(corrupt)
            predictor.predict = lambda image_path, top_k=1: {"predicted_class": "borek", "confidence": 0.01,
                                                              "top_predictions": [{"class_name": "borek", "probability": 0.01}]}
            low = pipeline.predict(path)
            self.assertEqual(low["confidence"], 0.01)
            self.assertEqual(low["nutrients"], result["nutrients"])
            with self.assertRaises(ValueError):
                estimate("unknown_regional_food", catalog=self.catalog)

    def test_real_phase5_checkpoint_provenance_when_available(self):
        checkpoint_path = ROOT / "ml/artifacts/efficientnet_b0_phase5_best.pt"
        if not checkpoint_path.is_file():
            self.skipTest("A trained Phase 5 checkpoint is not present")
        _, metadata = load_checkpoint(checkpoint_path, expected_classes=PHASE5_CLASSES)
        training = metadata["training"]
        self.assertEqual(training["epoch"], 5)
        self.assertIn("Phase 3 EfficientNet-B0 features only", training["initialization"])
        self.assertEqual(training["class_weighting"], "inverse-frequency weights N/(13*n_class), normalized to mean 1")
        self.assertEqual(training["phase3_checkpoint_sha256"], digest(ROOT / "ml/artifacts/efficientnet_b0_best.pt"))
        self.assertEqual(metadata["classes"], list(PHASE5_CLASSES))
        self.assertEqual(metadata["class_to_idx"], {name: index for index, name in enumerate(PHASE5_CLASSES)})
        phase3 = torch.load(ROOT / "ml/artifacts/efficientnet_b0_best.pt", map_location="cpu", weights_only=True)
        phase5 = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        phase3_features = {name for name in phase3["model_state_dict"] if name.startswith("features.")}
        self.assertTrue(phase3_features)
        for name in phase3_features:
            self.assertEqual(phase3["model_state_dict"][name].shape, phase5["model_state_dict"][name].shape)
        initialized = create_model(ModelConfig(num_classes=13, pretrained=False))
        initialize_from_phase3(initialized, ROOT / "ml/artifacts/efficientnet_b0_best.pt")
        initialized_state = initialized.state_dict()
        for name in phase3_features:
            self.assertTrue(torch.equal(phase3["model_state_dict"][name], initialized_state[name]), name)
        self.assertEqual(phase5["model_state_dict"]["classifier.1.weight"].shape[0], 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
