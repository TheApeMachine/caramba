"""Scorecard

Computes simple objective scores for CCP demo runs.
For the Unknown Format Decoder party trick, the key score is decode accuracy
against Lab ground truth and tool test pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Scorecard.

    Holds scalar metrics for a run step or a full run.
    """

    decode_ok: bool
    tool_tests_ok: bool
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "decode_ok": bool(self.decode_ok),
            "tool_tests_ok": bool(self.tool_tests_ok),
            "details": dict(self.details),
        }

