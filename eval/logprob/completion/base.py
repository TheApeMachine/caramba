"""Completion log-probability scoring protocol."""

from __future__ import annotations

from typing import Protocol


class LogprobCompletion(Protocol):
    """Scores a completion continuation given tokenized prompt and completion."""

    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float: ...

