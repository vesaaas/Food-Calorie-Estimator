"""Meal logging service that reuses the existing nutrition estimator."""

from datetime import date, datetime, timezone
import uuid

from nutrition.src.calculator import estimate

from .models import DiaryState, MealEntry
from .repositories import DiaryRepository


class MealLogService:
    def __init__(self, repository: DiaryRepository, *, catalog=None, clock=None):
        self.repository = repository
        self.catalog = catalog
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def add_meal(
        self, food_id: str, *, grams=None, servings=None,
        logged_at: datetime | None = None, classifier_confidence: float | None = None,
        meal_id: str | None = None,
    ) -> MealEntry:
        calculated = estimate(food_id, grams=grams, servings=servings, catalog=self.catalog)
        timestamp = self.clock() if logged_at is None else logged_at
        entry = MealEntry(
            meal_id=str(uuid.uuid4()) if meal_id is None else meal_id,
            food_id=food_id, display_name=calculated["display_name"], logged_at=timestamp,
            portion_grams=calculated["portion_grams"], portion_mode=calculated["portion_mode"],
            calories_kcal=calculated["nutrients"]["energy_kcal"],
            protein_g=calculated["nutrients"]["protein_g"],
            carbohydrates_g=calculated["nutrients"]["carbohydrates_g"],
            fat_g=calculated["nutrients"]["fat_g"],
            classifier_confidence=classifier_confidence,
        )
        state = self.repository.load()
        if any(existing.meal_id == entry.meal_id for existing in state.meals):
            raise ValueError(f"Duplicate meal_id: {entry.meal_id}")
        self.repository.save(DiaryState((*state.meals, entry), state.daily_calorie_goal))
        return entry

    def list_meals(self) -> tuple[MealEntry, ...]:
        return self.repository.load().meals

    def meals_for_day(self, day, *, timezone_info=timezone.utc) -> tuple[MealEntry, ...]:
        if isinstance(day, datetime) or not isinstance(day, date):
            raise ValueError("day must be a datetime.date")
        return tuple(
            meal for meal in self.list_meals()
            if meal.logged_at.astimezone(timezone_info).date() == day
        )
