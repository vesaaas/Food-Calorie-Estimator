"""Personal calorie-goal service, backed by the replaceable diary repository."""

from .models import DiaryState, finite_positive
from .repositories import DiaryRepository


class CalorieGoalService:
    def __init__(self, repository: DiaryRepository):
        self.repository = repository

    def get_daily_goal(self) -> float | None:
        return self.repository.load().daily_calorie_goal

    def set_daily_goal(self, calories: float) -> float:
        goal = finite_positive(calories, "daily_calorie_goal")
        state = self.repository.load()
        self.repository.save(DiaryState(state.meals, goal))
        return goal

    def clear_daily_goal(self) -> None:
        state = self.repository.load()
        self.repository.save(DiaryState(state.meals, None))
