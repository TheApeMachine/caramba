"""Completion log-probability scoring protocol."""

from __future__ import annotations

from typing import Protocol


class LogprobCompletion(Protocol):
    """Scores a completion continuation given tokenized prompt and completion."""

    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float: ...

    def score_batch(
        self, *, prompt_ids: list[int], completions_ids: list[list[int]]
    ) -> list[float]:
        """Batch score multiple completions (optional optimization).

        If not implemented, scorer will fall back to sequential scoring.
        """
        ...

