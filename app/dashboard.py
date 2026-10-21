"""Pure daily/weekly aggregation and repository-backed dashboard queries."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
import math

from .models import MealEntry
from .repositories import DiaryRepository


@dataclass(frozen=True)
class DailySummary:
    day: date
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    calorie_goal_kcal: float | None
    remaining_calories_kcal: float | None
    progress_fraction: float | None
    goal_exceeded: bool
    meal_count: int


@dataclass(frozen=True)
class WeeklySummary:
    start_date: date
    end_date: date
    days: tuple[DailySummary, ...]
    total_calories_kcal: float
    average_daily_calories_kcal: float
    total_protein_g: float
    total_carbohydrates_g: float
    total_fat_g: float
    calorie_goal_kcal_per_day: float | None
    weekly_calorie_goal_kcal: float | None
    remaining_calories_kcal: float | None
    progress_fraction: float | None
    goal_exceeded: bool


def _safe_total(values) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError("Dashboard total is not finite") from error
    if not math.isfinite(total):
        raise ValueError("Dashboard total is not finite")
    return total


def summarize_day(day: date, meals: tuple[MealEntry, ...] | list[MealEntry], goal: float | None) -> DailySummary:
    if isinstance(day, datetime) or not isinstance(day, date):
        raise ValueError("day must be a datetime.date")
    if goal is not None and (type(goal) not in (int, float) or not math.isfinite(goal) or goal <= 0):
        raise ValueError("goal must be finite and positive, or None")
    if any(not isinstance(meal, MealEntry) for meal in meals):
        raise ValueError("meals must contain only MealEntry records")
    calories = _safe_total(meal.calories_kcal for meal in meals)
    protein = _safe_total(meal.protein_g for meal in meals)
    carbohydrates = _safe_total(meal.carbohydrates_g for meal in meals)
    fat = _safe_total(meal.fat_g for meal in meals)
    remaining = None if goal is None else float(goal) - calories
    progress = None if goal is None else calories / float(goal)
    if progress is not None and not math.isfinite(progress):
        raise ValueError("Daily progress is not finite")
    return DailySummary(
        day=day, calories_kcal=calories, protein_g=protein,
        carbohydrates_g=carbohydrates, fat_g=fat,
        calorie_goal_kcal=None if goal is None else float(goal),
        remaining_calories_kcal=remaining, progress_fraction=progress,
        goal_exceeded=False if goal is None else calories > float(goal),
        meal_count=len(meals),
    )


def summarize_week(
    ending_on: date, meals: tuple[MealEntry, ...] | list[MealEntry], goal: float | None,
    *, timezone_info: tzinfo = timezone.utc,
) -> WeeklySummary:
    """Summarize the inclusive seven-day calendar window ending_on−6 through ending_on."""
    if isinstance(ending_on, datetime) or not isinstance(ending_on, date):
        raise ValueError("ending_on must be a datetime.date")
    start = ending_on - timedelta(days=6)
    by_day: dict[date, list[MealEntry]] = {}
    for meal in meals:
        local_day = meal.logged_at.astimezone(timezone_info).date()
        if start <= local_day <= ending_on:
            by_day.setdefault(local_day, []).append(meal)
    days = tuple(
        summarize_day(start + timedelta(days=offset), by_day.get(start + timedelta(days=offset), []), goal)
        for offset in range(7)
    )
    total_calories = _safe_total(day.calories_kcal for day in days)
    total_protein = _safe_total(day.protein_g for day in days)
    total_carbohydrates = _safe_total(day.carbohydrates_g for day in days)
    total_fat = _safe_total(day.fat_g for day in days)
    weekly_goal = None if goal is None else float(goal) * 7
    if weekly_goal is not None and not math.isfinite(weekly_goal):
        raise ValueError("weekly goal is not finite")
    remaining = None if weekly_goal is None else weekly_goal - total_calories
    progress = None if weekly_goal is None else total_calories / weekly_goal
    if progress is not None and not math.isfinite(progress):
        raise ValueError("Weekly progress is not finite")
    return WeeklySummary(
        start_date=start, end_date=ending_on, days=days,
        total_calories_kcal=total_calories,
        average_daily_calories_kcal=total_calories / 7,
        total_protein_g=total_protein, total_carbohydrates_g=total_carbohydrates,
        total_fat_g=total_fat, calorie_goal_kcal_per_day=None if goal is None else float(goal),
        weekly_calorie_goal_kcal=weekly_goal, remaining_calories_kcal=remaining,
        progress_fraction=progress,
        goal_exceeded=False if weekly_goal is None else total_calories > weekly_goal,
    )


class DashboardService:
    def __init__(self, repository: DiaryRepository, *, timezone_info: tzinfo = timezone.utc, clock=None):
        self.repository = repository
        self.timezone_info = timezone_info
        self.clock = clock or (lambda: datetime.now(timezone_info))

    def daily_summary(self, day: date | None = None) -> DailySummary:
        state = self.repository.load()
        current_day = day or self.clock().astimezone(self.timezone_info).date()
        entries = [
            meal for meal in state.meals
            if meal.logged_at.astimezone(self.timezone_info).date() == current_day
        ]
        return summarize_day(current_day, entries, state.daily_calorie_goal)

    def weekly_summary(self, ending_on: date | None = None) -> WeeklySummary:
        state = self.repository.load()
        end = ending_on or self.clock().astimezone(self.timezone_info).date()
        return summarize_week(end, state.meals, state.daily_calorie_goal, timezone_info=self.timezone_info)
