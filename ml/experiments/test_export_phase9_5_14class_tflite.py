"""Fast synthetic checks for Phase 9.5 export/evaluation helpers."""

import unittest

import numpy as np

import export_phase9_5_14class_tflite as export


CLASSES = tuple([f"food_{i}" for i in range(13)] + ["non_food"])


class ExportHelperTests(unittest.TestCase):
    def test_deterministic_sample_covers_all_triage_groups_and_classes(self):
        rows = []
        for label in CLASSES[:13]:
            for i in range(3):
                rows.append({"split": "validation", "triage_label": export.SUPPORTED,
                             "food_class": label, "source_dataset": "synthetic",
                             "source_id": f"{label}-{i}"})
        for group in (export.NON_FOOD, export.UNSUPPORTED):
            for i in range(4):
                rows.append({"split": "validation", "triage_label": group,
                             "food_class": "pear" if group == export.UNSUPPORTED else "",
                             "source_dataset": "synthetic", "source_id": f"{group}-{i}"})
        one = export.deterministic_validation_sample(rows, CLASSES, 2, 3)
        two = export.deterministic_validation_sample(rows, CLASSES, 2, 3)
        self.assertEqual([r["source_id"] for r in one], [r["source_id"] for r in two])
        self.assertEqual(len(one), 32)
        self.assertEqual(sum(r["triage_label"] == export.SUPPORTED for r in one), 26)
        self.assertEqual(sum(r["triage_label"] == export.NON_FOOD for r in one), 3)
        self.assertEqual(sum(r["triage_label"] == export.UNSUPPORTED for r in one), 3)

    def test_argmax_metrics_separate_nonfood_and_unsupported(self):
        rows = [{"triage_label": export.SUPPORTED, "food_class": label} for label in CLASSES[:13]]
        rows.extend([
            {"triage_label": export.NON_FOOD, "food_class": ""},
            {"triage_label": export.UNSUPPORTED, "food_class": "apple"},
        ])
        probabilities = np.concatenate((np.eye(14, dtype=np.float32)[:13], np.asarray([
            [0.8] + [0.0] * 12 + [0.2],
            [0.9] + [0.0] * 12 + [0.1],
        ], dtype=np.float32)))
        result = export._model_metrics(rows, probabilities, CLASSES)
        self.assertEqual(result["supported_food_acceptance"], 1.0)
        self.assertEqual(result["non_food_rejection"], 0.0)
        self.assertEqual(result["unsupported_food_rejection"], 0.0)
        self.assertEqual(result["false_counts"]["non_food_false_acceptances"], 1)
        self.assertEqual(result["false_counts"]["unsupported_food_false_acceptances"], 1)


if __name__ == "__main__":
    unittest.main()
