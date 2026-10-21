"""Reusable food-diary domain services for the future mobile application."""

from .dashboard import DashboardService, DailySummary, WeeklySummary
from .goals import CalorieGoalService
from .meal_log import MealLogService
from .models import DiaryState, MealEntry
from .repositories import DiaryRepository
from .storage import JsonDiaryRepository, PersistenceError

__all__ = [
    "CalorieGoalService", "DashboardService", "DiaryRepository", "DiaryState",
    "JsonDiaryRepository", "MealEntry", "MealLogService", "PersistenceError",
    "DailySummary", "WeeklySummary",
]
