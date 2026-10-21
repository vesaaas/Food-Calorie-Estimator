"""Storage boundary used by application/domain services."""

from typing import Protocol

from .models import DiaryState


class DiaryRepository(Protocol):
    """Replaceable storage contract; services depend on this, not on JSON or paths."""

    def load(self) -> DiaryState: ...

    def save(self, state: DiaryState) -> None: ...
