"""Log-probability scorer.

Orchestrates tokenization + choice selection, delegating completion scoring to a
composed completion scorer specialization.
"""
from __future__ import annotations

from caramba.data.tokenizers.base import Tokenizer
from caramba.eval.logprob.completion.base import LogprobCompletion


class LogprobScorer:
    """Scores completions and choices via next-token log-probabilities."""
    def __init__(self, *, tokenizer: Tokenizer, completion: LogprobCompletion) -> None:
        self.tokenizer = tokenizer
        self.completion = completion

    def score_completion_logprob(self, *, prompt: str, completion: str) -> float:
        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)
        if not prompt_ids or not completion_ids:
            # If encoding collapses (rare), make it very unlikely.
            return float("-inf")
        return float(self.completion.score(prompt_ids=prompt_ids, completion_ids=completion_ids))

    def pick_choice_by_logprob(self, *, prompt: str, choices: list[str]) -> str:
        if not choices:
            raise ValueError("choices must be non-empty")
        best: tuple[float, str] | None = None
        for c in choices:
            s = self.score_completion_logprob(prompt=prompt, completion=str(c))
            item = (float(s), str(c))
            best = item if best is None or item[0] > best[0] else best
        assert best is not None
        return best[1]

    def pick_choice_with_scores(
        self, *, prompt: str, choices: list[str]
    ) -> tuple[str, list[tuple[str, float]]]:
        if not choices:
            raise ValueError("choices must be non-empty")
        scored: list[tuple[str, float]] = []
        for c in choices:
            s = self.score_completion_logprob(prompt=prompt, completion=str(c))
            scored.append((str(c), float(s)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0], scored

