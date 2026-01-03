"""Event primitives.

The project is moving toward an event-native external interface:
- The atomic interaction unit is an EventEnvelope (JSON contract).
- Internally, models can still operate on discrete steps (tokens) as VM time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import time
import uuid


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """A minimal JSON-serializable event envelope.

    Required fields (v0):
    - type:     event type identifier (e.g. "Message", "ToolResult", "Wake", "Idle")
    - payload:  JSON-serializable payload
    - sender:   stable sender identity (agent/persona/user id)

    Optional fields (v0):
    - priority: higher is more urgent (default 0)
    - budget_ms: optional compute/latency budget for handling this event
    - id: unique event id (generated if absent)
    - ts: unix timestamp seconds (generated if absent)
    """

    type: str
    payload: Any
    sender: str
    priority: int = 0
    budget_ms: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": str(self.id),
            "ts": float(self.ts),
            "type": str(self.type),
            "sender": str(self.sender),
            "priority": int(self.priority),
            "payload": self.payload,
        }
        if self.budget_ms is not None:
            d["budget_ms"] = int(self.budget_ms)
        return d

    @staticmethod
    def from_json_dict(obj: Mapping[str, Any]) -> "EventEnvelope":
        if not isinstance(obj, Mapping):
            raise TypeError(f"Expected Mapping for EventEnvelope, got {type(obj).__name__}")

        missing = [k for k in ("type", "payload", "sender") if k not in obj]
        if missing:
            raise KeyError(f"Missing required EventEnvelope fields: {missing}")

        etype = obj["type"]
        sender = obj["sender"]
        payload = obj["payload"]
        if not isinstance(etype, str) or not etype.strip():
            raise ValueError("EventEnvelope.type must be a non-empty string")
        if not isinstance(sender, str) or not sender.strip():
            raise ValueError("EventEnvelope.sender must be a non-empty string")

        priority = obj.get("priority", 0)
        budget_ms = obj.get("budget_ms", None)
        eid = obj.get("id", uuid.uuid4().hex)
        ts = obj.get("ts", time.time())

        try:
            priority_i = int(priority)
        except Exception as e:
            raise TypeError(f"EventEnvelope.priority must be int-like, got {priority!r}") from e

        budget_i: int | None
        if budget_ms is None:
            budget_i = None
        else:
            try:
                budget_i = int(budget_ms)
            except Exception as e:
                raise TypeError(f"EventEnvelope.budget_ms must be int-like, got {budget_ms!r}") from e
            if budget_i < 0:
                raise ValueError(f"EventEnvelope.budget_ms must be >= 0, got {budget_i}")

        if not isinstance(eid, str) or not eid.strip():
            raise ValueError("EventEnvelope.id must be a non-empty string")
        try:
            ts_f = float(ts)
        except Exception as e:
            raise TypeError(f"EventEnvelope.ts must be float-like, got {ts!r}") from e

        return EventEnvelope(
            type=etype.strip(),
            payload=payload,
            sender=sender.strip(),
            priority=priority_i,
            budget_ms=budget_i,
            id=eid.strip(),
            ts=ts_f,
        )

