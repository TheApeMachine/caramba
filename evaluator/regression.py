"""Regression suite

Defines a deterministic non-regression harness for CCP runs.
This is a thin runner that can replay fixed traces and compare expected outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.runtime.trace.reader import TraceReader
from caramba.runtime.trace.schema import TraceEvent


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """One regression case definition."""

    trace_path: Path
    expected_kinds: list[str]


@dataclass(slots=True)
class RegressionSuite:
    """Regression suite runner."""

    cases: list[RegressionCase]

    def run(self) -> dict[str, Any]:
        """Run all regression cases and return a report."""
        outputs: list[dict[str, Any]] = []
        for case in self.cases:
            outputs.append(self.run_case(case))
        ok = all(bool(x.get("ok")) for x in outputs)
        return {"ok": bool(ok), "cases": outputs}

    def run_case(self, case: RegressionCase) -> dict[str, Any]:
        if not isinstance(case, RegressionCase):
            raise TypeError(f"case must be RegressionCase, got {type(case).__name__}")
        reader = TraceReader(path=case.trace_path)
        kinds = [ev.kind for ev in reader.events()]
        ok = kinds == list(case.expected_kinds)
        return {"ok": bool(ok), "trace": str(case.trace_path), "kinds": kinds, "expected": list(case.expected_kinds)}

