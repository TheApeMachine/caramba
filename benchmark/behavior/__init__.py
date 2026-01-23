from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.behavior.benchmark import BehaviorBenchmark
from benchmark.behavior.types import (
    BehaviorResult,
    BehaviorSummary,
    CaseResult,
    GeneratedCase,
    MatchType,
    Difficulty,
    EvalKind,
)

__all__ = [
    "BehaviorBenchmark",
    "BehaviorResult",
    "BehaviorSummary",
    "CaseResult",
    "GeneratedCase",
    "MatchType",
    "Difficulty",
    "EvalKind",
]


def __getattr__(name: str) -> Any:
    if name == "BehaviorBenchmark":
        from benchmark.behavior.benchmark import BehaviorBenchmark as _BehaviorBenchmark

        return _BehaviorBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
