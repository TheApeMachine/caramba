"""Trace schema

Defines the canonical event record format used for deterministic tracing and replay.
The goal is to capture every externally relevant decision (events, tool results,
verdicts, seeds) so runs can be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Trace event record.

    This is the smallest durable unit in a run trace. It must be JSON-serializable.
    """

    kind: str
    payload: dict[str, Any]
    ts: float

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON object.

        Used to write newline-delimited JSON trace files.
        """
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("TraceEvent.kind must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise TypeError(f"TraceEvent.payload must be a dict, got {type(self.payload).__name__}")
        return {"kind": str(self.kind), "payload": self.payload, "ts": float(self.ts)}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "TraceEvent":
        """Parse a TraceEvent from a JSON object."""
        if not isinstance(obj, dict):
            raise TypeError(f"TraceEvent JSON must be a dict, got {type(obj).__name__}")
        kind = obj.get("kind")
        payload = obj.get("payload")
        ts = obj.get("ts")
        if not isinstance(kind, str) or not kind:
            raise ValueError("TraceEvent.kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError(f"TraceEvent.payload must be a dict, got {type(payload).__name__}")
        if not isinstance(ts, (int, float)):
            raise TypeError(f"TraceEvent.ts must be number-like, got {type(ts).__name__}")
        return TraceEvent(kind=str(kind), payload=dict(payload), ts=float(ts))

