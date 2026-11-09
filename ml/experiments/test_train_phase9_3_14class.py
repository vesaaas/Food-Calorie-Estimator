"""Synthetic smoke tests for the Phase 9.3 isolated 14-class runner."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import tensorflow as tf

import train_phase9_3_14class as runner


class Phase93RunnerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.classes = tuple(runner.load_config()["classes"])

    @staticmethod
    def _tiny_13class_model() -> tf.keras.Model:
        image = tf.keras.Input((224, 224, 3), name="image_rgb_0_255")
        backbone_input = tf.keras.Input((224, 224, 3))
        x = tf.keras.layers.Conv2D(2, 1, use_bias=False, name="tiny_conv")(backbone_input)
        x = tf.keras.layers.BatchNormalization(name="tiny_batch_norm")(x)
        x = tf.keras.layers.Activation("relu", name="tiny_activation_1")(x)
        x = tf.keras.layers.Activation("relu", name="tiny_activation_2")(x)
        x = tf.keras.layers.Activation("relu", name="tiny_activation_3")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        backbone = tf.keras.Model(backbone_input, x, name="MobileNetV3Small")
        features = backbone(image)
        features = tf.keras.layers.Dropout(0.2, name="head_dropout")(features)
        output = tf.keras.layers.Dense(13, activation="softmax", name="class_probabilities")(features)
        return tf.keras.Model(image, output, name="synthetic_13class")

    def test_mapping_and_head_replacement_preserve_backbone(self):
        config = runner.load_config()
        self.assertEqual(config["classes"][:13], list(self.classes[:13]))
        self.assertEqual(config["class_to_idx"]["non_food"], 13)
        base = self._tiny_13class_model()
        before = [weight.copy() for weight in base.get_layer("MobileNetV3Small").get_weights()]
        candidate = runner.build_14class_model(base, self.classes, augmentation=False)
        self.assertEqual(candidate.output_shape, (None, 14))
        self.assertEqual(candidate.get_layer("phase9_3_class_probabilities").units, 14)
        self.assertFalse(candidate.get_layer("phase9_3_frozen_feature_extractor").trainable)
        self.assertTrue(candidate.get_layer("phase9_3_class_probabilities").trainable)
        after = candidate.get_layer("phase9_3_frozen_feature_extractor").get_layer(
            "MobileNetV3Small"
        ).get_weights()
        for prior, current in zip(before, after, strict=True):
            np.testing.assert_array_equal(prior, current)

    def test_optional_fine_tune_freezes_batch_normalization(self):
        candidate = runner.build_14class_model(
            self._tiny_13class_model(), self.classes, augmentation=False,
        )
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "head_best.keras"
            candidate.save(checkpoint)
            fine_model, selected, checks = runner.prepare_finetune_model(
                checkpoint, self.classes,
            )
        backbone = fine_model.get_layer("phase9_3_frozen_feature_extractor").get_layer(
            "MobileNetV3Small"
        )
        self.assertFalse(backbone.get_layer("tiny_batch_norm").trainable)
        self.assertTrue(any(layer.trainable for layer in backbone.layers
                            if layer.name in selected))
        self.assertTrue(checks["batch_normalization_frozen"])
        self.assertEqual(checks["trainable_backbone_layer_count"], len(selected))
        self.assertTrue(fine_model.get_layer("phase9_3_class_probabilities").trainable)

    def test_production_preprocessing_shape_dtype_and_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.jpg"
            pixels = np.zeros((300, 400, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(400, dtype=np.uint8)[None, :]
            Image.fromarray(pixels).save(path)
            result = runner.preprocess_image(path)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 255.0)

    def test_loader_ignores_test_paths_and_keeps_unsupported_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "tiny.jpg"
            Image.new("RGB", (8, 8), (40, 80, 120)).save(image_path)
            manifest = root / "manifest.csv"
            fields = [
                "path", "split", "triage_label", "food_class", "source_dataset",
                "source_id", "group_id", "duplicate_label_conflict", "sha256", "dhash",
            ]
            rows = []
            serial = 0
            for split, prefix in (("train", "0"), ("validation", "f")):
                for label in self.classes[:-1]:
                    rows.append(self._row(
                        image_path, split, runner.SUPPORTED, label, "Food-101", serial,
                        dhash=prefix * 16,
                    ))
                    serial += 1
                rows.append(self._row(
                    image_path, split, runner.NON_FOOD, "", "Open Images V7", serial,
                    dhash=prefix * 16,
                ))
                serial += 1
            rows.append(self._row(
                image_path, "validation", runner.UNSUPPORTED, "churros", "Food-101", serial,
                dhash="f" * 16,
            ))
            serial += 1
            rows.append(self._row(
                root / "does-not-exist-test.jpg", "test", runner.SUPPORTED, "pizza",
                "Food-101", serial, dhash="a" * 16,
            ))
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            partitions = runner.load_fit_and_validation_rows(manifest, self.classes[:-1])
        self.assertEqual(len(partitions["train_fit"]), 14)
        self.assertEqual(len(partitions["validation_fit"]), 14)
        self.assertEqual(len(partitions["validation_unsupported"]), 1)
        self.assertNotIn("does-not-exist-test.jpg", str(partitions))
        self.assertNotIn(runner.UNSUPPORTED, [row["label"] for row in partitions["train_fit"]])

    def test_acceptance_and_classification_accuracy_are_separate(self):
        rows = [
            {"triage_label": runner.SUPPORTED, "food_class": label}
            for label in self.classes[:-1]
        ] + [
            {"triage_label": runner.NON_FOOD, "food_class": ""},
            {"triage_label": runner.UNSUPPORTED, "food_class": "churros"},
        ]
        probabilities = np.zeros((len(rows), 14), dtype=np.float32)
        for index in range(13):
            probabilities[index, (index + 1) % 13] = 0.9
        probabilities[13, 13] = 0.9
        probabilities[14, 13] = 0.9
        metrics = runner.validation_metrics(rows, probabilities, self.classes, 1.0)
        gate = metrics["acceptance_and_rejection"]
        self.assertEqual(gate["supported_food_acceptance_overall"], 1.0)
        self.assertEqual(gate["supported_classification_accuracy"], 0.0)
        self.assertEqual(gate["non_food_rejection"], 1.0)
        self.assertEqual(gate["unsupported_food_rejection"], 1.0)
        self.assertFalse(gate["meets_targets"])
        self.assertIsNone(metrics["decision_policy"]["confidence_threshold"])

    def test_fit_stage_saves_synthetic_checkpoint_and_log(self):
        inputs = tf.keras.Input((14,), name="synthetic_features")
        outputs = tf.keras.layers.Dense(14, activation="softmax")(inputs)
        model = tf.keras.Model(inputs, outputs)
        x = np.eye(14, dtype=np.float32)
        y = np.arange(14, dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(7)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "tiny_best.keras"
            history = Path(temp) / "tiny_history.csv"
            result = runner.fit_stage(
                model, dataset, dataset, checkpoint_path=checkpoint,
                history_path=history, epochs=1, learning_rate=1e-3, patience=0,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(history.is_file())
            self.assertEqual(result["epochs_completed"], 1)
            restored = tf.keras.models.load_model(checkpoint, compile=False)
            self.assertEqual(restored.output_shape, (None, 14))

    def test_finetune_resume_plan_uses_recorded_head_checkpoint_and_new_output(self):
        config = runner.load_config()
        with tempfile.TemporaryDirectory() as temp:
            experiment_root = Path(temp) / "artifacts" / "experiments"
            head_dir = experiment_root / "phase9_3_14class"
            head_dir.mkdir(parents=True)
            checkpoint = head_dir / "phase9_3_14class_head_best.keras"
            checkpoint.write_bytes(b"synthetic checkpoint marker")
            head_report_path = head_dir / "training_report.json"
            head_report = {
                "experiment": config["experiment"],
                "status": "validation_targets_not_met",
                "classes": config["classes"],
                "class_to_idx": config["class_to_idx"],
                "selected_checkpoint": str(checkpoint),
                "stages": {
                    "frozen_head": {
                        "checkpoint": str(checkpoint),
                        "validation": {"acceptance_and_rejection": {}},
                    },
                },
                "validation": {"acceptance_and_rejection": {}},
                "held_out_test": {"evaluated": False},
            }
            head_report_path.write_text(json.dumps(head_report), encoding="utf-8")
            output = experiment_root / "phase9_3_14class_finetune"
            resumed, report_path, separate_output, parsed_report = runner.resolve_finetune_resume(
                config,
                resume_checkpoint=checkpoint,
                head_report_path=head_report_path,
                output_dir=output,
                experiment_root=experiment_root,
            )
            self.assertEqual(resumed, checkpoint.resolve())
            self.assertEqual(report_path, head_report_path.resolve())
            self.assertEqual(separate_output, output.resolve())
            self.assertEqual(parsed_report["selected_checkpoint"], str(checkpoint))
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(head_report_path.is_file())
            self.assertFalse(output.exists())

    def test_finetune_resume_rejects_existing_or_mismatched_inputs(self):
        config = runner.load_config()
        with tempfile.TemporaryDirectory() as temp:
            experiment_root = Path(temp) / "experiments"
            head_dir = experiment_root / "head"
            head_dir.mkdir(parents=True)
            checkpoint = head_dir / "best.keras"
            checkpoint.write_bytes(b"marker")
            head_report_path = head_dir / "report.json"
            head_report = {
                "experiment": config["experiment"],
                "classes": config["classes"],
                "class_to_idx": config["class_to_idx"],
                "selected_checkpoint": str(checkpoint),
                "stages": {"frozen_head": {
                    "checkpoint": str(checkpoint),
                    "validation": {"acceptance_and_rejection": {}},
                }},
                "held_out_test": {"evaluated": False},
            }
            head_report_path.write_text(json.dumps(head_report), encoding="utf-8")
            existing_output = experiment_root / "fine"
            existing_output.mkdir()
            with self.assertRaises(FileExistsError):
                runner.resolve_finetune_resume(
                    config, resume_checkpoint=checkpoint,
                    head_report_path=head_report_path,
                    output_dir=existing_output, experiment_root=experiment_root,
                )
            bad_report = dict(head_report)
            bad_report["class_to_idx"] = {**config["class_to_idx"], "non_food": 12}
            head_report_path.write_text(json.dumps(bad_report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class indices"):
                runner.resolve_finetune_resume(
                    config, resume_checkpoint=checkpoint,
                    head_report_path=head_report_path,
                    output_dir=experiment_root / "another-fine",
                    experiment_root=experiment_root,
                )

    def test_finetune_only_plan_does_not_call_training(self):
        config = runner.load_config()
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "head.keras"
            checkpoint.write_bytes(b"synthetic checkpoint marker")
            report_path = Path(temp) / "head-report.json"
            output = Path(temp) / "new-fine-output"
            mocked_report = {"status": "validation_targets_not_met"}
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", [
                "train_phase9_3_14class.py", "--fine-tune-only",
            ]), mock.patch.object(runner, "load_config", return_value=config), mock.patch.object(
                runner, "resolve_finetune_resume",
                return_value=(checkpoint, report_path, output, mocked_report),
            ), mock.patch.object(
                runner, "prepare_finetune_model",
                return_value=(None, ["tiny_conv"], {"batch_normalization_frozen": True}),
            ), mock.patch.object(runner, "fit_stage") as fit_stage, redirect_stdout(stdout):
                runner.main()
            plan = json.loads(stdout.getvalue())
            self.assertFalse(plan["training_started"])
            self.assertFalse(plan["head_only_stage_repeated"])
            self.assertEqual(plan["resume_checkpoint"], str(checkpoint))
            fit_stage.assert_not_called()

    @staticmethod
    def _row(path, split, triage, food_class, source, serial, *, dhash):
        return {
            "path": str(path), "split": split, "triage_label": triage,
            "food_class": food_class, "source_dataset": source,
            "source_id": str(serial), "group_id": f"{source}:{serial}",
            "duplicate_label_conflict": "false",
            "sha256": f"{serial + 1:064x}", "dhash": dhash,
        }


if __name__ == "__main__":
    unittest.main()
