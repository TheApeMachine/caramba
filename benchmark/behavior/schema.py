from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from benchmark.behavior.types import ContentConstraint, Difficulty, EvalKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlotSpec(StrictModel):
    """How to generate a single templating slot value."""

    # Exactly one of: pool, ref, calc.
    # Draw a value from a named suite pool.
    pool: str | None = None
    # Derive from another slot (must be defined earlier in the same template instance).
    ref: str | None = None
    # Compute from other slots (must be defined earlier in the same template instance).
    calc: dict[str, Any] | None = None
    # Ensure uniqueness across slots sharing this key (within one case instance).
    unique_key: str | None = None
    # Optional post-processing.
    transform: Literal["upper", "lower", "title", "none"] = "none"

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        n = int(bool(self.pool)) + int(bool(self.ref)) + int(bool(self.calc))
        if n != 1:
            raise ValueError("SlotSpec must set exactly one of: pool, ref, calc.")


class ChoiceExplicitSpec(StrictModel):
    mode: Literal["explicit"] = "explicit"
    choices: list[str]
    # If true, choices will be shuffled deterministically and the expected answer
    # will be tracked via `correct_index`.
    shuffle: bool = True


class ChoiceFromPoolSpec(StrictModel):
    mode: Literal["from_pool"] = "from_pool"
    pool: str
    num_choices: PositiveInt = 4
    # Correct option (templated string, usually a slot like "${target}").
    correct: str
    # If true, shuffle deterministically.
    shuffle: bool = True


ChoiceSpec = ChoiceExplicitSpec | ChoiceFromPoolSpec


class TemplateSpec(StrictModel):
    """A case template; expanded `repeat` times into concrete cases."""

    id: str
    repeat: PositiveInt = 1

    difficulty: Difficulty
    kind: EvalKind

    prompt: str
    expected: str

    # Optional: specify a target substring expected to be present in the prompt,
    # used for attention highlight spans. If unset, we default to `expected`.
    target_text: str | None = None

    # Optional multiple-choice config (required for CHOICE_LOGPROB).
    choice: ChoiceSpec | None = None

    # Scoring policy for CONTAINED credit
    allow_contained: bool = False
    contained_constraints: list[ContentConstraint] = Field(default_factory=list)
    disallow_contained_if_expected_in_prompt: bool = True

    # Slot generators used by prompt/expected/choices.
    slots: dict[str, SlotSpec] = Field(default_factory=dict)

    # Extra per-case metadata (must be JSON-serializable).
    metadata: dict[str, Any] = Field(default_factory=dict)


class CategorySpec(StrictModel):
    id: str
    description: str = ""
    templates: list[TemplateSpec]


class BehaviorSuiteSpec(StrictModel):
    """YAML spec for the unified behavior benchmark suite."""

    version: PositiveInt = 1
    id: str = "behavior"
    seed: PositiveInt

    # How many cases per category after expansion.
    tests_per_category: PositiveInt = 30

    # Pools used by SlotSpec.
    pools: dict[str, list[str]] = Field(default_factory=dict)

    # Difficulty weights applied after raw match score.
    difficulty_weights: dict[Difficulty, float] = Field(
        default_factory=lambda: {
            Difficulty.EASY: 1.0,
            Difficulty.MEDIUM: 1.25,
            Difficulty.HARD: 1.5,
        }
    )

    # Baseline-relative weighting multipliers. Keys are model match type, then baseline match type.
    # (These weights multiply the raw score after difficulty weighting.)
    baseline_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            # Model EXACT
            "exact": {"exact": 0.5, "contained": 0.75, "none": 1.0},
            # Model CONTAINED
            "contained": {"exact": 0.25, "contained": 0.5, "none": 0.75},
            # Model NONE
            "none": {"exact": 0.0, "contained": 0.0, "none": 0.0},
        }
    )

    categories: list[CategorySpec]

