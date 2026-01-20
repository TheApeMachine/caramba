"""Log-probability scorer.

Orchestrates tokenization + choice selection, delegating completion scoring to a
composed completion scorer specialization.

OPTIMIZED: Added batched scoring for multiple choices to reduce forward passes.
"""
from __future__ import annotations

from typing import Protocol, cast

from data.tokenizers.base import Tokenizer
from eval.logprob.completion.base import LogprobCompletion


class LogprobCompletionWithBatch(Protocol):
    """Protocol for LogprobCompletion with batch scoring support."""
    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float: ...
    def score_batch(self, *, prompt_ids: list[int], completions_ids: list[list[int]]) -> list[float]: ...


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
        """Pick the best choice and return all scores.

        OPTIMIZED: Uses batched scoring when completion scorer supports it.
        """
        if not choices:
            raise ValueError("choices must be non-empty")

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            # Fallback for empty prompt
            return choices[0], [(c, float("-inf")) for c in choices]

        # Try batched scoring if available
        if hasattr(self.completion, 'score_batch'):
            return self._pick_choice_batched(prompt_ids, choices)

        # Fallback to sequential scoring
        scored: list[tuple[str, float]] = []
        for c in choices:
            completion_ids = self.tokenizer.encode(str(c))
            if not completion_ids:
                scored.append((str(c), float("-inf")))
            else:
                s = float(self.completion.score(prompt_ids=prompt_ids, completion_ids=completion_ids))
                scored.append((str(c), s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0], scored

    def _pick_choice_batched(
        self, prompt_ids: list[int], choices: list[str]
    ) -> tuple[str, list[tuple[str, float]]]:
        """Batched choice scoring using completion.score_batch()."""
        # Tokenize all choices
        choice_ids_list = []
        valid_choices = []
        for c in choices:
            completion_ids = self.tokenizer.encode(str(c))
            if completion_ids:
                choice_ids_list.append(completion_ids)
                valid_choices.append(str(c))

        if not choice_ids_list:
            return choices[0], [(c, float("-inf")) for c in choices]

        # Batch score
        # Type checker doesn't recognize Protocol methods even after hasattr check
        # We know score_batch exists because hasattr check passed, so we cast
        completion_with_batch = cast('LogprobCompletionWithBatch', self.completion)
        scores = completion_with_batch.score_batch(prompt_ids=prompt_ids, completions_ids=choice_ids_list)

        # Build scored list
        scored: list[tuple[str, float]] = []
        score_idx = 0
        for c in choices:
            if str(c) in valid_choices:
                scored.append((str(c), float(scores[score_idx])))
                score_idx += 1
            else:
                scored.append((str(c), float("-inf")))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0], scored

