"""Atomic JSON persistence adapter for the Phase 6 diary repository contract."""

import json
from pathlib import Path
import tempfile

from .models import DiaryState, MealEntry

DEFAULT_DIARY_PATH = Path(__file__).resolve().parent / "data" / "diary.json"
STORAGE_SCHEMA_VERSION = 1


class PersistenceError(ValueError):
    """Stored diary data exists but cannot be safely loaded or written."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise PersistenceError(f"Nonfinite JSON constant: {value}")


class JsonDiaryRepository:
    """Persist a validated DiaryState; missing storage represents an empty diary."""

    def __init__(self, path: Path = DEFAULT_DIARY_PATH):
        self.path = Path(path).expanduser().resolve()

    def load(self) -> DiaryState:
        if not self.path.exists():
            return DiaryState()
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                parse_constant=_reject_constant,
            )
            if (not isinstance(payload, dict)
                    or type(payload.get("schema_version")) is not int
                    or payload["schema_version"] != STORAGE_SCHEMA_VERSION):
                raise PersistenceError("Unsupported or missing diary schema_version")
            if set(payload) != {"schema_version", "meals", "daily_calorie_goal"}:
                raise PersistenceError("Diary file has missing or unexpected top-level fields")
            if not isinstance(payload["meals"], list):
                raise PersistenceError("Diary meals must be an array")
            meals = tuple(MealEntry.from_record(record) for record in payload["meals"])
            return DiaryState(meals=meals, daily_calorie_goal=payload["daily_calorie_goal"])
        except PersistenceError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise PersistenceError(f"Could not load diary file '{self.path}': {error}") from error

    def save(self, state: DiaryState) -> None:
        if not isinstance(state, DiaryState):
            raise TypeError("state must be a DiaryState")
        temporary = None
        try:
            payload = {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "meals": [meal.to_record() for meal in state.meals],
                "daily_calorie_goal": state.daily_calorie_goal,
            }
            serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
            temporary.replace(self.path)
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, PersistenceError):
                raise
            raise PersistenceError(f"Could not save diary file '{self.path}': {error}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
