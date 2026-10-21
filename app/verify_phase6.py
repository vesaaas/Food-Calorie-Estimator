"""Offline verification for Phase 6 domain logic and persistence."""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from nutrition.src.calculator import estimate
from nutrition.src.catalog import load_catalog

from .dashboard import DashboardService, summarize_day, summarize_week
from .goals import CalorieGoalService
from .meal_log import MealLogService
from .models import DiaryState, MealEntry
from .storage import JsonDiaryRepository, PersistenceError


class MemoryRepository:
    """Small test adapter demonstrating services do not depend on JSON files."""

    def __init__(self, state=None):
        self.state = state or DiaryState()

    def load(self):
        return self.state

    def save(self, state):
        self.state = state


class Phase6Verification(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.repository = MemoryRepository()
        self.now = datetime(2026, 5, 20, 12, 30, tzinfo=timezone.utc)
        self.meals = MealLogService(
            self.repository, catalog=self.catalog, clock=lambda: self.now,
        )

    def test_default_serving_meal_reuses_nutrition_estimator(self):
        entry = self.meals.add_meal("pizza")
        result = estimate("pizza", catalog=self.catalog)
        self.assertEqual(entry.portion_mode, "default")
        self.assertEqual(entry.portion_grams, result["portion_grams"])
        self.assertEqual(entry.calories_kcal, result["nutrients"]["energy_kcal"])

    def test_servings_multiplier_and_custom_grams(self):
        serving = estimate("borek", servings=1.5, catalog=self.catalog)
        entry = self.meals.add_meal("borek", servings=1.5)
        self.assertEqual(entry.portion_mode, "servings")
        self.assertEqual(entry.portion_grams, serving["portion_grams"])
        custom = self.meals.add_meal("kebap", grams=180)
        expected = estimate("kebap", grams=180, catalog=self.catalog)
        self.assertEqual(custom.portion_mode, "grams")
        self.assertEqual(custom.nutrients, expected["nutrients"])

    def test_meal_round_trip_through_json_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonDiaryRepository(Path(directory) / "nested" / "diary.json")
            self.assertEqual(repository.load(), DiaryState())
            service = MealLogService(repository, catalog=self.catalog, clock=lambda: self.now)
            original = service.add_meal("sushi", servings=2, classifier_confidence=0.72)
            loaded = JsonDiaryRepository(repository.path).load()
            self.assertEqual(loaded.meals, (original,))
            self.assertEqual(loaded.meals[0].classifier_confidence, 0.72)

    def test_multiple_meals_and_nutrient_aggregation_by_day(self):
        first = self.meals.add_meal("pizza", logged_at=self.now)
        second = self.meals.add_meal("sushi", grams=160, logged_at=self.now + timedelta(hours=1))
        self.meals.add_meal("steak", logged_at=self.now + timedelta(days=1))
        summary = DashboardService(self.repository).daily_summary(self.now.date())
        self.assertEqual(summary.meal_count, 2)
        self.assertAlmostEqual(summary.calories_kcal, first.calories_kcal + second.calories_kcal)
        self.assertAlmostEqual(summary.protein_g, first.protein_g + second.protein_g)
        self.assertAlmostEqual(summary.carbohydrates_g, first.carbohydrates_g + second.carbohydrates_g)
        self.assertAlmostEqual(summary.fat_g, first.fat_g + second.fat_g)

    def test_daily_goal_can_be_set_updated_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonDiaryRepository(Path(directory) / "diary.json")
            goals = CalorieGoalService(repository)
            self.assertIsNone(goals.get_daily_goal())
            self.assertEqual(goals.set_daily_goal(2000), 2000)
            goals.set_daily_goal(2150.5)
            self.assertEqual(JsonDiaryRepository(repository.path).load().daily_calorie_goal, 2150.5)

    def test_invalid_goals_rejected_without_mutation(self):
        goals = CalorieGoalService(self.repository)
        for invalid in (0, -1, True, "2000", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                goals.set_daily_goal(invalid)
        self.assertIsNone(goals.get_daily_goal())

    def test_invalid_portions_rejected(self):
        for kwargs in (
            {"grams": 0}, {"grams": -1}, {"grams": True}, {"grams": float("nan")},
            {"grams": float("inf")}, {"servings": 0}, {"servings": -1},
            {"servings": float("nan")}, {"servings": float("inf")},
            {"grams": 100, "servings": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.meals.add_meal("pizza", **kwargs)
        with self.assertRaises(ValueError):
            self.meals.add_meal("not_a_food")

    def test_malformed_persisted_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diary.json"
            cases = [
                "{broken",
                '{"schema_version":1,"schema_version":1,"meals":[],"daily_calorie_goal":null}',
                json.dumps({"schema_version": 1, "meals": [{"food_id": "pizza"}], "daily_calorie_goal": None}),
                json.dumps({"schema_version": 1, "meals": [], "daily_calorie_goal": float("nan")}),
                json.dumps({"schema_version": 99, "meals": [], "daily_calorie_goal": None}),
            ]
            for contents in cases:
                path.write_text(contents, encoding="utf-8")
                with self.subTest(contents=contents), self.assertRaises(PersistenceError):
                    JsonDiaryRepository(path).load()

    def test_daily_goal_progress_exceeded_and_no_goal(self):
        entry = self.meals.add_meal("pizza", grams=100, logged_at=self.now)
        below = summarize_day(self.now.date(), [entry], entry.calories_kcal + 50)
        self.assertAlmostEqual(below.remaining_calories_kcal, 50)
        self.assertAlmostEqual(below.progress_fraction, entry.calories_kcal / (entry.calories_kcal + 50))
        self.assertFalse(below.goal_exceeded)
        above = summarize_day(self.now.date(), [entry], entry.calories_kcal / 2)
        self.assertTrue(above.goal_exceeded)
        self.assertLess(above.remaining_calories_kcal, 0)
        self.assertGreater(above.progress_fraction, 1)
        none = summarize_day(self.now.date(), [entry], None)
        self.assertIsNone(none.calorie_goal_kcal)
        self.assertIsNone(none.remaining_calories_kcal)
        self.assertIsNone(none.progress_fraction)
        self.assertFalse(none.goal_exceeded)

    def test_week_is_inclusive_seven_calendar_days_and_empty_days_are_zero(self):
        end = self.now.date()
        start = end - timedelta(days=6)
        first = self.meals.add_meal("pizza", logged_at=datetime.combine(start, datetime.min.time(), timezone.utc))
        last = self.meals.add_meal("sarma", logged_at=datetime.combine(end, datetime.min.time(), timezone.utc))
        self.meals.add_meal("steak", logged_at=datetime.combine(start - timedelta(days=1), datetime.min.time(), timezone.utc))
        week = summarize_week(end, self.repository.load().meals, 2000)
        self.assertEqual((week.start_date, week.end_date), (start, end))
        self.assertEqual(len(week.days), 7)
        self.assertEqual([day.day for day in week.days], [start + timedelta(days=i) for i in range(7)])
        self.assertEqual(week.days[0].meal_count, 1)
        self.assertEqual(week.days[0].calories_kcal, first.calories_kcal)
        self.assertEqual(week.days[-1].meal_count, 1)
        self.assertEqual(week.days[-1].calories_kcal, last.calories_kcal)
        self.assertTrue(all(day.calories_kcal == 0 for day in week.days[1:-1]))
        self.assertEqual(week.total_calories_kcal, first.calories_kcal + last.calories_kcal)

    def test_confidence_does_not_change_nutrition_or_portion(self):
        low = self.meals.add_meal("sarma", grams=125, classifier_confidence=0.01)
        high = self.meals.add_meal("sarma", grams=125, classifier_confidence=0.99)
        self.assertEqual(low.portion_grams, high.portion_grams)
        self.assertEqual(low.nutrients, high.nutrients)

    def test_thirteen_phase5_labels_have_nutrition_and_existing_classes_still_calculate(self):
        expected = (
            "pizza", "hamburger", "sushi", "steak", "ice_cream", "donuts", "omelette",
            "french_fries", "chicken_wings", "spaghetti_bolognese", "borek", "kebap", "sarma",
        )
        self.catalog.validate_coverage(expected)
        self.assertEqual(set(self.catalog.food_ids), set(expected))
        for food_id in expected:
            with self.subTest(food_id=food_id):
                self.assertGreater(estimate(food_id, catalog=self.catalog)["portion_grams"], 0)

    def test_timezone_aware_day_aggregation(self):
        day = date(2026, 5, 20)
        self.meals.add_meal("pizza", logged_at=datetime(2026, 5, 20, 23, 30, tzinfo=timezone.utc))
        local = timezone(timedelta(hours=2))
        summary = DashboardService(self.repository, timezone_info=local).daily_summary(day + timedelta(days=1))
        self.assertEqual(summary.meal_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
