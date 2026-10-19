"""Offline Phase 4 checks; no ML imports, model execution, or Food-101 access."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if __package__:
    from .catalog import DEFAULT_CATALOG, NUTRIENTS, NutritionCatalog, load_catalog
    from .calculator import estimate, scale_nutrients
else:
    from catalog import DEFAULT_CATALOG, NUTRIENTS, NutritionCatalog, load_catalog
    from calculator import estimate, scale_nutrients

ROOT = Path(__file__).resolve().parents[2]


def classifier_labels() -> tuple[str, ...]:
    """Read the literal ML class tuple without importing torch or executing ML code."""
    tree = ast.parse((ROOT / "ml/src/dataset.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEVELOPMENT_CLASSES"
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError("DEVELOPMENT_CLASSES literal not found")


class NutritionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DEFAULT_CATALOG.read_text())
        cls.catalog = load_catalog(expected_classes=classifier_labels())

    def fixture(self):
        """Synthetic arithmetic values, confined to a copy in test memory."""
        data = deepcopy(self.data)
        food = data["foods"][0]
        food["per_100g"] = dict(zip(NUTRIENTS, (200, 10, 20, 8)))
        food["default_serving"]["grams"] = 150
        return NutritionCatalog(data)

    def test_exact_ten_class_coverage(self):
        labels = classifier_labels()
        self.assertEqual(len(labels), 10)
        self.assertEqual(set(labels), set(self.catalog.food_ids))
        for label in labels:
            self.assertEqual(self.catalog.lookup(label)["food_id"], label)
        for expected in (labels[:-1], (*labels, "local_dish"), (*labels, labels[0]), "pizza", [None]):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.catalog.validate_coverage(expected)

    def test_lookup_ignores_json_order(self):
        data = deepcopy(self.data)
        data["foods"].reverse()
        reordered = NutritionCatalog(data, expected_classes=classifier_labels())
        for label in self.catalog.food_ids:
            self.assertEqual(reordered.lookup(label), self.catalog.lookup(label))

    def test_unknown_and_nonexact_labels(self):
        for label in ("Pizza", "pizza ", "ice cream", "local_dish", "", 0, None, []):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Unknown food_id"):
                self.catalog.lookup(label)

    def test_100g_reproduces_every_reference(self):
        for label in self.catalog.food_ids:
            self.assertEqual(estimate(label, grams=100, catalog=self.catalog)["nutrients"],
                             self.catalog.lookup(label)["per_100g"])

    def test_default_and_explicit_grams(self):
        catalog = self.fixture()
        default = estimate("pizza", catalog=catalog)
        self.assertEqual(default["portion_mode"], "default")
        self.assertEqual(default["portion_grams"], 150)
        self.assertEqual(default["nutrients"], dict(zip(NUTRIENTS, (300, 15, 30, 12))))
        explicit = estimate("pizza", grams=180, catalog=catalog)
        self.assertEqual(explicit["portion_mode"], "grams")
        self.assertEqual(explicit["nutrients"], dict(zip(NUTRIENTS, (360, 18, 36, 14.4))))
        for label in self.catalog.food_ids:
            grams = self.catalog.lookup(label)["default_serving"]["grams"]
            self.assertEqual(estimate(label, catalog=self.catalog)["nutrients"],
                             estimate(label, grams=grams, catalog=self.catalog)["nutrients"])

    def test_fractional_and_multiple_servings(self):
        for servings, grams, calories in ((0.5, 75, 150), (1.5, 225, 450), (2, 300, 600)):
            with self.subTest(servings=servings):
                result = estimate("pizza", servings=servings, catalog=self.fixture())
                self.assertEqual(result["portion_grams"], grams)
                self.assertEqual(result["nutrients"]["energy_kcal"], calories)
                self.assertEqual(result["portion_mode"], "servings")

    def test_invalid_portions_and_overflow(self):
        for field in ("grams", "servings"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "100", [], 10**400):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    estimate("pizza", catalog=self.catalog, **{field: value})
        with self.assertRaises(ValueError):
            estimate("pizza", grams=100, servings=1, catalog=self.catalog)
        with self.assertRaises(ValueError):
            estimate("pizza", servings=1e308, catalog=self.catalog)
        with self.assertRaises(ValueError):
            estimate("pizza", grams=1e308, catalog=self.catalog)

    def test_calculations_and_returned_data_do_not_mutate_catalog(self):
        data = deepcopy(self.data)
        catalog = NutritionCatalog(data)
        before = catalog.lookup("pizza")
        for _ in range(3):
            result = estimate("pizza", servings=1.3, catalog=catalog)
            result["default_serving"]["grams"] = 1
            result["nutrition_source"]["fdc_id"] = 1
        copied = catalog.lookup("pizza")
        copied["per_100g"]["energy_kcal"] = 1
        data["foods"][0]["per_100g"]["energy_kcal"] = 1
        self.assertEqual(catalog.lookup("pizza"), before)
        nutrients = dict(zip(NUTRIENTS, (200, 10, 20, 8)))
        original = nutrients.copy()
        self.assertAlmostEqual(scale_nutrients(nutrients, 123.456)["energy_kcal"], 246.912)
        self.assertEqual(nutrients, original)

    def test_missing_fields(self):
        paths = [(key,) for key in ("schema_version", "catalog_version", "foods")]
        food = self.data["foods"][0]
        paths += [("foods", 0, key) for key in food]
        for section in ("per_100g", "default_serving", "nutrition_source"):
            paths += [("foods", 0, section, key) for key in food[section]]
        for path in paths:
            data = deepcopy(self.data)
            target = data
            for part in path[:-1]:
                target = target[part]
            del target[path[-1]]
            with self.subTest(path=path), self.assertRaises(ValueError):
                NutritionCatalog(data)

    def test_duplicate_ids_and_invalid_structure(self):
        data = deepcopy(self.data)
        data["foods"].append(deepcopy(data["foods"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate food_id"):
            NutritionCatalog(data)
        for invalid in (None, [], {}, {**self.data, "schema_version": True},
                        {**self.data, "schema_version": 2}, {**self.data, "foods": []},
                        {**self.data, "foods": [None]}, {**self.data, "foods": {}}):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                NutritionCatalog(invalid)

    def test_invalid_nutrients_and_servings(self):
        for key in NUTRIENTS:
            for value in (None, -1, float("nan"), float("inf"), True, "10", 10**400):
                data = deepcopy(self.data)
                data["foods"][0]["per_100g"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    NutritionCatalog(data)
        for value in (0, -1, float("nan"), float("inf"), True, "100"):
            data = deepcopy(self.data)
            data["foods"][0]["default_serving"]["grams"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                NutritionCatalog(data)

    def test_invalid_provenance(self):
        for key, value in (("fdc_id", True), ("provider", "unknown"), ("data_type", "unknown"),
                           ("record_description", "different food"), ("retrieved_on", "yesterday"),
                           ("url", "https://example.com/food"), ("download_url", ""),
                           ("url", "https://fdc.nal.usda.gov/food-details/1/nutrients")):
            data = deepcopy(self.data)
            data["foods"][0]["nutrition_source"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                NutritionCatalog(data)

    def test_json_loading_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            with self.assertRaises(ValueError):
                load_catalog(path)
            for content in ("", "{", '{"schema_version":1,"schema_version":1}',
                            '{"foods": NaN}', '{"foods": Infinity}',
                            json.dumps(self.data).replace('"energy_kcal": 271', '"energy_kcal": 1e999')):
                path.write_text(content)
                with self.subTest(content=content[:40]), self.assertRaises(ValueError):
                    load_catalog(path)

    def test_production_matches_offline_usda_evidence(self):
        proof = json.loads((DEFAULT_CATALOG.parent / "usda_source_extracts.json").read_text())
        records = {record["fdcId"]: record for record in proof["records"]}
        self.assertEqual(len(records), 10)
        names = {1008: "energy_kcal", 1003: "protein_g", 1005: "carbohydrates_g", 1004: "fat_g"}
        for food_id in self.catalog.food_ids:
            food = self.catalog.lookup(food_id)
            source = food["nutrition_source"]
            record = records[source["fdc_id"]]
            self.assertEqual(record["description"], source["record_description"])
            self.assertEqual(record["dataType"], source["data_type"])
            self.assertEqual(source["retrieved_on"], proof["retrieved_on"])
            self.assertIn(source["download_url"], [a["url"] for a in proof["archives"]])
            values = {}
            for nutrient in record["foodNutrients"]:
                nutrient_id = nutrient["nutrient"]["id"]
                self.assertEqual(nutrient["nutrient"]["unitName"], "kcal" if nutrient_id == 1008 else "g")
                values[names[nutrient_id]] = nutrient["amount"]
            self.assertEqual(values, food["per_100g"])
            portion, = record["foodPortions"]
            self.assertEqual(portion["id"], food["default_serving"]["source_portion_id"])
            self.assertEqual(portion["gramWeight"], food["default_serving"]["grams"])
            label = portion.get("portionDescription", "1 " + portion.get("modifier", ""))
            self.assertEqual(label, food["default_serving"]["label"])

    def test_cli_and_offline_public_import_from_other_directory(self):
        cli = ROOT / "nutrition/src/estimate.py"
        with tempfile.TemporaryDirectory() as directory:
            for args, expected in ((["pizza"], "357.72 kcal"),
                                   (["pizza", "--grams", "180"], "487.80 kcal"),
                                   (["pizza", "--servings", "1.5"], "536.58 kcal"),
                                   (["pizza", "--servings", "2"], "715.44 kcal")):
                result = subprocess.run([sys.executable, str(cli), *args], cwd=directory,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
            for args in (["unknown"], ["pizza", "--grams", "0"],
                         ["pizza", "--grams", "nan"], ["pizza", "--grams", "100", "--servings", "1"]):
                result = subprocess.run([sys.executable, str(cli), *args], cwd=directory,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
        code = "import sys; from nutrition.src import estimate; assert estimate('pizza')['food_id']=='pizza'; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
