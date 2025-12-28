"""Paper review types.

This module intentionally stays lightweight: it provides the *types* used by the
research loop and manifests, without depending on any reviewer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ReviewConfig:
    """Configuration for review runs (AI-assisted or manual).

    This is intentionally minimal; the research loop can extend it without
    impacting training.
    """

    enabled: bool = True
    max_iterations: int = 3
    model: str | None = None
    extra: dict[str, Any] | None = None


class Recommendation(str, Enum):
    """High-level recommendation for the paper."""

    APPROVE = "approve"
    MINOR_REVISIONS = "minor_revisions"
    MAJOR_REVISIONS = "major_revisions"
    REJECT = "reject"


@dataclass(frozen=True)
class ProposedExperiment:
    """A suggested follow-up experiment."""

    name: str
    rationale: str = ""


@dataclass(frozen=True)
class ReviewResult:
    """Result of a review pass.

    Note: The research loop expects these attributes (e.g. `overall_score`),
    regardless of which reviewer implementation produced the object.
    """

    overall_score: float
    recommendation: Recommendation = Recommendation.MAJOR_REVISIONS

    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    proposed_experiments: list[ProposedExperiment] = field(default_factory=list)

    # Convenience/compat for older call sites
    approved: bool = False
    notes: str = ""
    style_fixes_only: bool = False

    @property
    def score(self) -> float:
        """Alias for `overall_score` (kept for backward compatibility)."""

        return self.overall_score

