"""Synthetic metric tests; these never load models or held-out images."""

import unittest

import numpy as np

import evaluate_phase9_4_14class_test as evaluation


CLASSES = tuple([f"food_{i}" for i in range(13)] + ["non_food"])


class FrozenHeldoutMetricTests(unittest.TestCase):
    def test_argmax_metrics_and_group_confusion_are_separate(self):
        rows = [
            {"triage_label": evaluation.SUPPORTED, "food_class": label}
            for label in CLASSES[:13]
        ] + [
            {"triage_label": evaluation.NON_FOOD, "food_class": ""},
            {"triage_label": evaluation.UNSUPPORTED, "food_class": "apple"},
        ]
        candidate = np.concatenate((np.eye(14, dtype=np.float32)[:13], np.asarray([
            [0.9] + [0.0] * 12 + [0.1],
            [0.0] * 13 + [1.0],
        ], dtype=np.float32)), axis=0)
        # Supported rows are correctly classified except class 0; the last
        # supported class is an explicit non_food false rejection.
        candidate[0] = np.eye(14, dtype=np.float32)[1]
        candidate[12] = np.eye(14, dtype=np.float32)[13]
        candidate[14] = np.eye(14, dtype=np.float32)[0]
        production = np.eye(13, dtype=np.float32)
        result = evaluation.metrics(rows, candidate, production, CLASSES)

        self.assertEqual(result["supported_food_acceptance_overall"], 12 / 13)
        self.assertEqual(result["supported_food_by_class"]["food_12"]["rejected_as_non_food"], 1)
        self.assertEqual(result["non_food_rejection"], 0.0)
        self.assertEqual(result["unsupported_food_rejection"], 0.0)
        self.assertEqual(result["supported_food_top1_accuracy_14class"], 11 / 13)
        self.assertEqual(result["effective_accepted_and_correct_accuracy"], 11 / 13)
        self.assertEqual(result["false_counts"]["non_food_false_acceptances_as_supported_food"], 1)
        self.assertEqual(result["false_counts"]["unsupported_food_false_acceptances_as_supported_food"], 1)
        self.assertEqual(len(result["confusion_matrix"]["counts"]), 15)
        self.assertEqual(len(result["confusion_matrix"]["counts"][0]), 14)

    def test_predeclared_targets_are_reported_without_policy_tuning(self):
        metrics = {
            "supported_food_acceptance_overall": 0.98,
            "supported_food_acceptance_each_class_min": 0.94,
            "non_food_rejection": 0.9,
            "unsupported_food_rejection": 0.1,
            "classification_accuracy_drop_absolute_vs_keras": 0.01,
            "supported_food_by_class": {"pizza": {"acceptance": 0.94}},
        }
        config = {"acceptance_targets": {
            "source": "Project engineering targets, not reviewer-specified requirements.",
            "supported_food_acceptance_overall_min": 0.97,
            "supported_food_acceptance_each_class_min": 0.95,
            "non_food_rejection_min": 0.80,
            "unsupported_food_rejection_min": 0.50,
            "supported_classification_accuracy_max_drop_absolute": 0.02,
        }}
        target_report = evaluation.acceptance_targets(config, metrics)
        self.assertFalse(target_report["all_met"])
        self.assertFalse(target_report["met"]["unsupported_food_rejection_min"])
        self.assertFalse(target_report["class_acceptance_all_classes_at_least_min"])


if __name__ == "__main__":
    unittest.main()
