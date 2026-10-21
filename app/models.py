"""Validated domain records; independent of storage format and UI frameworks."""

from dataclasses import dataclass
from datetime import datetime
import math
import re

NUTRIENT_FIELDS = ("calories_kcal", "protein_g", "carbohydrates_g", "fat_g")
PORTION_MODES = {"default", "grams", "servings"}


def finite_nonnegative(value, field: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return float(value)


def finite_positive(value, field: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class MealEntry:
    """One persisted meal and the nutrition result calculated for its chosen portion."""

    meal_id: str
    food_id: str
    display_name: str
    logged_at: datetime
    portion_grams: float
    portion_mode: str
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    classifier_confidence: float | None = None

    def __post_init__(self):
        if not isinstance(self.meal_id, str) or not self.meal_id.strip():
            raise ValueError("meal_id must be a nonempty string")
        if not isinstance(self.food_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.food_id):
            raise ValueError("food_id must be an exact canonical food identifier")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be a nonempty string")
        if not isinstance(self.logged_at, datetime) or self.logged_at.tzinfo is None or self.logged_at.utcoffset() is None:
            raise ValueError("logged_at must be a timezone-aware datetime")
        finite_positive(self.portion_grams, "portion_grams")
        if self.portion_mode not in PORTION_MODES:
            raise ValueError(f"portion_mode must be one of {sorted(PORTION_MODES)}")
        for name in NUTRIENT_FIELDS:
            finite_nonnegative(getattr(self, name), name)
        if self.classifier_confidence is not None:
            confidence = self.classifier_confidence
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("classifier_confidence must be finite and between 0 and 1")

    @property
    def nutrients(self) -> dict[str, float]:
        return {
            "energy_kcal": float(self.calories_kcal), "protein_g": float(self.protein_g),
            "carbohydrates_g": float(self.carbohydrates_g), "fat_g": float(self.fat_g),
        }

    def to_record(self) -> dict:
        return {
            "meal_id": self.meal_id, "food_id": self.food_id,
            "display_name": self.display_name, "logged_at": self.logged_at.isoformat(),
            "portion_grams": self.portion_grams, "portion_mode": self.portion_mode,
            "calories_kcal": self.calories_kcal, "protein_g": self.protein_g,
            "carbohydrates_g": self.carbohydrates_g, "fat_g": self.fat_g,
            "classifier_confidence": self.classifier_confidence,
        }

    @classmethod
    def from_record(cls, record: dict) -> "MealEntry":
        if not isinstance(record, dict):
            raise ValueError("meal record must be an object")
        required = {
            "meal_id", "food_id", "display_name", "logged_at", "portion_grams",
            "portion_mode", *NUTRIENT_FIELDS,
        }
        allowed = required | {"classifier_confidence"}
        missing, extra = required - record.keys(), record.keys() - allowed
        if missing or extra:
            raise ValueError(f"meal record fields invalid: missing={sorted(missing)}, unexpected={sorted(extra)}")
        try:
            logged_at = datetime.fromisoformat(record["logged_at"])
        except (TypeError, ValueError) as error:
            raise ValueError("logged_at must be an ISO-8601 timestamp") from error
        return cls(
            meal_id=record["meal_id"], food_id=record["food_id"],
            display_name=record["display_name"], logged_at=logged_at,
            portion_grams=record["portion_grams"], portion_mode=record["portion_mode"],
            calories_kcal=record["calories_kcal"], protein_g=record["protein_g"],
            carbohydrates_g=record["carbohydrates_g"], fat_g=record["fat_g"],
            classifier_confidence=record.get("classifier_confidence"),
        )


@dataclass(frozen=True)
class DiaryState:
    meals: tuple[MealEntry, ...] = ()
    daily_calorie_goal: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "meals", tuple(self.meals))
        if any(not isinstance(meal, MealEntry) for meal in self.meals):
            raise ValueError("meals must contain only MealEntry records")
        ids = [meal.meal_id for meal in self.meals]
        if len(ids) != len(set(ids)):
            raise ValueError("meal IDs must be unique")
        if self.daily_calorie_goal is not None:
            finite_positive(self.daily_calorie_goal, "daily_calorie_goal")
