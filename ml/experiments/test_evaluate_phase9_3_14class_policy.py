"""Lightweight synthetic tests for validation policy calibration utilities."""

import unittest

import numpy as np

import evaluate_phase9_3_14class_policy as evaluation


CLASSES = tuple([f"food_{i}" for i in range(13)] + ["non_food"])


def _rows():
    rows = []
    for index, label in enumerate(CLASSES[:-1]):
        rows.append({"triage_label": evaluation.SUPPORTED, "food_class": label,
                     "group_id": f"food-group-{index}", "path": f"food-{index}.jpg"})
    rows.extend(
        {"triage_label": evaluation.NON_FOOD, "food_class": "", "group_id": f"nf-{i}", "path": f"nf-{i}.jpg"}
        for i in range(20)
    )
    rows.extend(
        {"triage_label": evaluation.UNSUPPORTED, "food_class": "apple", "group_id": f"uf-{i}", "path": f"uf-{i}.jpg"}
        for i in range(20)
    )
    return rows


class DecisionPolicyTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        rows = _rows()
        cal_a, sel_a, report_a = evaluation.split_calibration_selection(rows)
        cal_b, sel_b, report_b = evaluation.split_calibration_selection(rows)
        self.assertEqual((cal_a, sel_a), (cal_b, sel_b))
        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a["group_overlap_count"], 0)
        cal_groups = {rows[i]["group_id"] for i in cal_a}
        sel_groups = {rows[i]["group_id"] for i in sel_a}
        self.assertFalse(cal_groups & sel_groups)

    def test_argmax_and_threshold_keep_unsupported_distinct(self):
        rows = _rows()
        probs = np.full((len(rows), 14), 0.001, dtype=np.float32)
        for i in range(13):
            probs[i, i] = 0.95
            probs[i, 13] = 0.01
        probs[13, 13], probs[14, 13] = 0.9, 0.1
        probs[33:, 13] = 0.8
        probs[33, 0], probs[33, 13] = 0.8, 0.05
        probs[34, 0], probs[34, 13] = 0.2, 0.7
        probs /= probs.sum(axis=1, keepdims=True)
        prod = np.zeros((len(rows), 13), dtype=np.float32)
        for i in range(13):
            prod[i, i] = 1.0
        prod[13:, 0] = 1.0

        argmax = evaluation.evaluate_policy(
            rows, probs, prod, CLASSES,
            {"kind": "argmax", "rule": "supported argmax"},
        )
        threshold = evaluation.evaluate_policy(
            rows, probs, prod, CLASSES,
            {"kind": "max_supported_probability", "threshold": 0.5},
        )
        self.assertEqual(argmax["samples"]["unsupported_food_evaluation_only"], 20)
        self.assertEqual(argmax["confusion_by_group"]["unsupported_food_evaluation_only"]["accepted_as_supported_food"], 1)
        self.assertEqual(threshold["confusion_by_group"]["unsupported_food_evaluation_only"]["accepted_as_supported_food"], 1)
        self.assertEqual(argmax["decision_metrics"]["supported_food_acceptance_overall"], 1.0)
        self.assertEqual(argmax["decision_metrics"]["effective_accepted_and_correct_classification_accuracy"], 1.0)

    def test_margin_and_class_conditioned_thresholds(self):
        probs = np.asarray([[0.70, 0.20, 0.10], [0.45, 0.40, 0.15]], dtype=np.float32)
        margin = evaluation._policy_acceptance(probs, {"kind": "prediction_margin", "threshold": 0.3})
        conditioned = evaluation._policy_acceptance(
            probs,
            {"kind": "class_conditioned_max_supported_probability", "thresholds": [0.65, 0.5]},
        )
        np.testing.assert_array_equal(margin, [True, False])
        np.testing.assert_array_equal(conditioned, [True, False])


if __name__ == "__main__":
    unittest.main()
