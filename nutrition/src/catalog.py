"""Validated local nutrition catalog; independent of the classifier and PyTorch."""

from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data" / "foods.json"
NUTRIENTS = ("energy_kcal", "protein_g", "carbohydrates_g", "fat_g")


def finite_number(value, field: str, *, positive: bool = False) -> float:
    """Reject booleans, numeric strings, nonfinite values, and invalid signs."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (value <= 0 if positive else value < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be a finite {bound} number")
    return float(value)


def _object(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(obj: dict, key: str, field: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}.{key} must be a nonempty string")
    return value


def _positive_id(obj: dict, key: str, field: str) -> None:
    if type(obj.get(key)) is not int or obj[key] <= 0:
        raise ValueError(f"{field}.{key} must be a positive integer")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


class NutritionCatalog:
    """Validate once; return copies so callers cannot mutate stored records."""

    def __init__(self, data: dict, *, expected_classes=None) -> None:
        data = deepcopy(_object(data, "catalog"))
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported or missing schema_version (expected 1)")
        self.catalog_version = _text(data, "catalog_version", "catalog")
        foods = data.get("foods")
        if not isinstance(foods, list) or not foods:
            raise ValueError("catalog.foods must be a nonempty array")
        self._foods = {}
        for position, food in enumerate(foods):
            field = f"foods[{position}]"
            _object(food, field)
            food_id = _text(food, "food_id", field)
            if not re.fullmatch(r"[a-z][a-z0-9_]*", food_id):
                raise ValueError(f"{field}.food_id must be an exact lowercase identifier")
            if food_id in self._foods:
                raise ValueError(f"Duplicate food_id: {food_id}")
            for key in ("display_name", "reference_description", "mapping_notes"):
                _text(food, key, field)
            nutrients = _object(food.get("per_100g"), f"{food_id}.per_100g")
            for key in NUTRIENTS:
                finite_number(nutrients.get(key), f"{food_id}.per_100g.{key}")
            serving = _object(food.get("default_serving"), f"{food_id}.default_serving")
            for key in ("label", "basis"):
                _text(serving, key, f"{food_id}.default_serving")
            finite_number(serving.get("grams"), f"{food_id}.default_serving.grams", positive=True)
            if serving.get("source_portion_id") is not None:
                _positive_id(serving, "source_portion_id", f"{food_id}.default_serving")
            source = _object(food.get("nutrition_source"), f"{food_id}.nutrition_source")
            for key in ("provider", "record_description", "url", "retrieved_on", "release"):
                _text(source, key, f"{food_id}.nutrition_source")
            if source["record_description"] != food["reference_description"]:
                raise ValueError(f"{food_id}: source and reference descriptions differ")
            url = urlparse(source["url"])
            if url.scheme != "https":
                raise ValueError(f"{food_id}: source URL must use HTTPS")
            if source["provider"] == "USDA FoodData Central":
                if serving.get("source_portion_id") is None:
                    raise ValueError(f"{food_id}: USDA default_serving requires source_portion_id")
                for key in ("data_type", "download_url"):
                    _text(source, key, f"{food_id}.nutrition_source")
                _positive_id(source, "fdc_id", f"{food_id}.nutrition_source")
                if source["data_type"] not in {"Survey (FNDDS)", "SR Legacy"}:
                    raise ValueError(f"{food_id}: unsupported USDA data type")
                archive = urlparse(source["download_url"])
                if archive.scheme != "https" or archive.netloc != "fdc.nal.usda.gov":
                    raise ValueError(f"{food_id}: download_url must reference official USDA FDC HTTPS")
                if url.netloc != "fdc.nal.usda.gov" or source["url"] != f"https://fdc.nal.usda.gov/food-details/{source['fdc_id']}/nutrients":
                    raise ValueError(f"{food_id}: source URL does not match official FDC record ID")
            elif source["provider"] == "TürKomp Turkish Food Composition Database":
                _text(source, "data_type", f"{food_id}.nutrition_source")
                _text(source, "source_record_id", f"{food_id}.nutrition_source")
                if url.netloc != "turkomp.tarimorman.gov.tr":
                    raise ValueError(f"{food_id}: TürKomp source URL must use its official domain")
            else:
                raise ValueError(f"{food_id}: unsupported nutrition provider {source['provider']!r}")
            try:
                if date.fromisoformat(source["retrieved_on"]).isoformat() != source["retrieved_on"]:
                    raise ValueError()
            except ValueError as error:
                raise ValueError(f"{food_id}: retrieved_on must be YYYY-MM-DD") from error
            self._foods[food_id] = food
        if expected_classes is not None:
            self.validate_coverage(expected_classes)

    @property
    def food_ids(self) -> tuple[str, ...]:
        return tuple(self._foods)

    def validate_coverage(self, expected_classes) -> None:
        """Require exact coverage; order is irrelevant and indices are not inferred."""
        if isinstance(expected_classes, str):
            raise ValueError("expected_classes must be a collection of class names")
        expected = tuple(expected_classes)
        if not expected or any(not isinstance(name, str) or not name for name in expected):
            raise ValueError("Expected class names must be nonempty strings")
        if len(set(expected)) != len(expected):
            raise ValueError("Expected class names must be unique")
        missing, extra = set(expected) - self._foods.keys(), self._foods.keys() - set(expected)
        if missing or extra:
            raise ValueError(f"Class coverage mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")

    def lookup(self, food_id: str) -> dict:
        if not isinstance(food_id, str) or food_id not in self._foods:
            raise ValueError(f"Unknown food_id: {food_id!r}; use an exact classifier label")
        return deepcopy(self._foods[food_id])


def load_catalog(path: Path = DEFAULT_CATALOG, *, expected_classes=None) -> NutritionCatalog:
    """Load local UTF-8 JSON, rejecting duplicate keys and nonfinite constants."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys,
                          parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load nutrition catalog '{path}': {error}") from error
    return NutritionCatalog(data, expected_classes=expected_classes)
