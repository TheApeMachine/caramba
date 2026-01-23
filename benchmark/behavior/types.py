from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvalKind(str, Enum):
    """How a case is evaluated."""

    # Greedy generation (next-token completion, deterministic).
    GENERATION_GREEDY = "generation_greedy"
    # Multiple-choice evaluated by teacher-forced logprob (deterministic).
    CHOICE_LOGPROB = "choice_logprob"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MatchType(str, Enum):
    NONE = "none"
    CONTAINED = "contained"
    EXACT = "exact"


class ContentConstraint(str, Enum):
    """Gates for awarding CONTAINED credit."""

    # Matching span must make up >= 50% of normalized output.
    MAJORITY = "MAJORITY"
    # Matching span must start the normalized output.
    EXACT_START = "EXACT_START"
    # Disallow soft credit if prompt+expected appears verbatim in output.
    NO_PARROTING = "NO_PARROTING"


@dataclass(frozen=True)
class GeneratedCase:
    """A fully-materialized, deterministic behavior case instance."""

    # Stable identifier (suite will prefix with category).
    id: str
    category: str
    difficulty: Difficulty
    kind: EvalKind

    prompt: str
    expected: str

    # For choice_logprob cases:
    choices: list[str] = field(default_factory=list)
    # Index of correct choice in `choices` (required if choices provided).
    correct_index: int | None = None

    # Scoring policy
    allow_contained: bool = False
    contained_constraints: list[ContentConstraint] = field(default_factory=list)
    # If true, disallow awarding CONTAINED when expected appears verbatim in prompt.
    disallow_contained_if_expected_in_prompt: bool = True

    # Optional "answer span" target for attention visualization.
    # If provided, we will attempt to find these tokens in the prompt and highlight.
    target_text: str | None = None

    # Arbitrary metadata (must be JSON-serializable).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceLogprobDetails:
    """Raw details for a choice_logprob evaluation (per model, per case)."""

    choices: list[str]
    logprobs: list[float]  # aligned with `choices`
    probs: list[float]  # softmax over logprobs (aligned with `choices`)
    picked: str
    picked_index: int
    correct: str
    correct_index: int
    margin_logprob: float  # logp(correct) - max(logp(other))


@dataclass(frozen=True)
class CaseModelOutput:
    """Per-model output + any raw scoring signals."""

    model_name: str
    output_text: str
    match_type: MatchType
    raw_score: float
    difficulty_weight: float
    baseline_weight: float
    final_score: float

    # Optional extra raw data
    choice_logprob: ChoiceLogprobDetails | None = None
    expected_logprob: float | None = None  # logprob(expected | prompt), when available


@dataclass(frozen=True)
class CaseResult:
    case: GeneratedCase
    outputs: dict[str, CaseModelOutput]  # model_name -> output/scoring


@dataclass(frozen=True)
class BehaviorSummary:
    model_name: str
    n: int
    exact: int
    contained: int
    none: int
    hard_accuracy: float
    soft_accuracy: float
    weighted_accuracy: float
    score_sum: float
    score_max: float

    # PPL-aware derived metrics (filled only if ppl data provided)
    ppl: float | None = None
    ppl_delta_vs_baseline: float | None = None
    ppl_adjusted_score: float | None = None


@dataclass(frozen=True)
class BehaviorResult:
    """Full behavior benchmark result for N models."""

    suite_id: str
    baseline_name: str
    cases: list[GeneratedCase]
    results: list[CaseResult]
    summaries: dict[str, BehaviorSummary]
    # Machine-readable config snapshot for auditability
    suite_config: dict[str, Any]

