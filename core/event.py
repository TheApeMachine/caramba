"""Event primitives.

The project is moving toward an event-native external interface:
- The atomic interaction unit is an EventEnvelope (binary contract).
- Internally, models can still operate on discrete steps (tokens) as VM time.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """A minimal binary event envelope.

    Required fields (v0):
    - type:     event type identifier (e.g. "Message", "ToolResult", "Wake", "Idle")
    - payload:  opaque bytes payload (typically a Cap'n Proto-encoded payload struct)
    - sender:   stable sender identity (agent/persona/user id)

    Optional fields (v0):
    - priority: higher is more urgent (default 0)
    - budget_ms: optional compute/latency budget for handling this event
    - commitment_delta: commitment lifecycle signal (-1 close, 0 neutral, +1 open)
    - commitment_id: optional id linking open/close pairs
    - id: unique event id (generated if absent)
    - ts: unix timestamp seconds (generated if absent)
    """

    type: str
    payload: bytes
    sender: str
    priority: int = 0
    budget_ms: int | None = None
    # Phase 2: commitment tracking
    commitment_delta: int = 0  # -1 close, 0 neutral, +1 open
    commitment_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        p = self.payload
        if isinstance(p, bytes):
            return
        if isinstance(p, bytearray):
            object.__setattr__(self, "payload", bytes(p))
            return
        if isinstance(p, memoryview):
            object.__setattr__(self, "payload", p.tobytes())
            return
        raise TypeError(
            "EventEnvelope.payload must be bytes-like (bytes|bytearray|memoryview). "
            f"Got {type(p).__name__}."
        )

    def to_json_dict(self) -> dict[str, Any]:
        """A JSON-safe representation for logs/metadata.

        Note: This is not the canonical event representation anymore. The canonical
        representation is binary: `EventEnvelope` + Cap'n Proto payload bytes.
        """
        cd = int(self.commitment_delta)
        if cd not in (-1, 0, 1):
            raise ValueError(f"EventEnvelope.commitment_delta must be in {{-1, 0, 1}}, got {cd}")
        d: dict[str, Any] = {
            "id": str(self.id),
            "ts": float(self.ts),
            "type": str(self.type),
            "sender": str(self.sender),
            "priority": int(self.priority),
            "payload_b64": base64.b64encode(self.payload).decode("ascii"),
            "commitment_delta": cd,
        }
        if self.budget_ms is not None:
            d["budget_ms"] = int(self.budget_ms)
        if self.commitment_id is not None:
            cid = self.commitment_id
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("EventEnvelope.commitment_id must be a non-empty string when provided")
            d["commitment_id"] = cid.strip()
        return d

    @staticmethod
    def from_json_dict(obj: Mapping[str, Any]) -> EventEnvelope:
        if not isinstance(obj, Mapping):
            raise TypeError(f"Expected Mapping for EventEnvelope, got {type(obj).__name__}")

        missing = [k for k in ("type", "payload_b64", "sender") if k not in obj]
        if missing:
            raise KeyError(f"Missing required EventEnvelope fields: {missing}")

        etype = obj["type"]
        sender = obj["sender"]
        payload_b64 = obj["payload_b64"]
        if not isinstance(etype, str) or not etype.strip():
            raise ValueError("EventEnvelope.type must be a non-empty string")
        if not isinstance(sender, str) or not sender.strip():
            raise ValueError("EventEnvelope.sender must be a non-empty string")
        if not isinstance(payload_b64, str):
            raise TypeError("EventEnvelope.payload_b64 must be a string")
        try:
            payload = base64.b64decode(payload_b64.encode("ascii"), validate=True)
        except Exception as e:
            raise ValueError("EventEnvelope.payload_b64 must be valid base64") from e

        priority = obj.get("priority", 0)
        budget_ms = obj.get("budget_ms", None)
        commitment_delta = obj.get("commitment_delta", 0)
        commitment_id = obj.get("commitment_id", None)
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

        try:
            cd_i = int(commitment_delta)
        except Exception as e:
            raise TypeError(
                f"EventEnvelope.commitment_delta must be int-like, got {commitment_delta!r}"
            ) from e
        if cd_i not in (-1, 0, 1):
            raise ValueError(f"EventEnvelope.commitment_delta must be in {{-1, 0, 1}}, got {cd_i}")

        cid_s: str | None
        if commitment_id is None:
            cid_s = None
        else:
            if not isinstance(commitment_id, str) or not commitment_id.strip():
                raise ValueError("EventEnvelope.commitment_id must be a non-empty string when provided")
            cid_s = commitment_id.strip()

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
            commitment_delta=cd_i,
            commitment_id=cid_s,
            id=eid.strip(),
            ts=ts_f,
        )

