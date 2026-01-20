"""Validity gate

Enforces basic correctness constraints:
- event payload must be JSON-serializable (for traceability)
- required keys must exist for certain event types (policy-specific)

This is intentionally strict: invalid data is rejected immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.event import EventEnvelope


@dataclass(slots=True)
class ValidityGate:
    """Validity gate.

    Used to validate incoming and outgoing EventEnvelope instances.
    """

    def validate_event(self, event: EventEnvelope) -> None:
        """Validate a single event."""
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"event must be EventEnvelope, got {type(event).__name__}")
        self.validate_json_serializable(event.to_json_dict())

    def validate_json_serializable(self, obj: Any) -> None:
        """Ensure an object is JSON-serializable."""
        try:
            json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception as e:
            raise TypeError(f"Object is not JSON-serializable: {type(e).__name__}: {e}") from e

