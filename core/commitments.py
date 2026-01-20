"""Commitment lifecycle primitives.

This module implements a minimal, runtime-side commitment ledger:
- Commitments are opened/closed via EventEnvelope meta-fields:
  - commitment_delta: +1 (open), 0 (neutral), -1 (close)
  - commitment_id: optional id linking open/close pairs

The ledger is architecture-agnostic: it can be used with any event-driven model/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import uuid

from core.event import EventEnvelope


@dataclass(frozen=True, slots=True)
class CommitmentOpen:
    sender: str
    ts: float


@dataclass(slots=True)
class CommitmentMetrics:
    opens: int = 0
    closes: int = 0
    close_without_id: int = 0
    idle_with_open_commitments: int = 0
    commitment_open_to_close_latency_s_sum: float = 0.0
    commitment_open_to_close_latency_s_count: int = 0

    def mean_open_to_close_latency_s(self) -> float | None:
        n = int(self.commitment_open_to_close_latency_s_count)
        if n <= 0:
            return None
        return float(self.commitment_open_to_close_latency_s_sum) / float(n)


@dataclass(slots=True)
class CommitmentLedger:
    """Tracks open commitments and exposes simple metrics."""

    metrics: CommitmentMetrics = field(default_factory=CommitmentMetrics)
    _open: dict[str, CommitmentOpen] = field(default_factory=dict)
    _open_order: list[str] = field(default_factory=list)  # insertion order for "most recent"

    def open_ids(self) -> list[str]:
        return list(self._open_order)

    def apply(self, event: EventEnvelope) -> EventEnvelope:
        """Update ledger state based on an event.

        Returns:
            The event, potentially rewritten to include an auto-generated or inferred commitment_id.
        """
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"event must be an EventEnvelope, got {type(event).__name__}")

        cd = int(event.commitment_delta)
        if cd not in (-1, 0, 1):
            raise ValueError(f"commitment_delta must be in {{-1,0,1}}, got {cd}")

        if str(event.type) == "Idle" and bool(self._open):
            self.metrics.idle_with_open_commitments += 1

        if cd == 0:
            return event

        if cd == +1:
            cid = event.commitment_id
            if cid is None:
                cid = uuid.uuid4().hex
                event = replace(event, commitment_id=cid)
            if cid in self._open:
                raise ValueError(f"commitment_id already open: {cid!r}")
            self._open[cid] = CommitmentOpen(sender=str(event.sender), ts=float(event.ts))
            self._open_order.append(cid)
            self.metrics.opens += 1
            return event

        # cd == -1
        cid_close = event.commitment_id
        if cid_close is None:
            cid_close = self._close_most_recent_open_for_sender(sender=str(event.sender))
            event = replace(event, commitment_id=cid_close)
            self.metrics.close_without_id += 1

        opened = self._open.get(cid_close)
        if opened is None:
            raise ValueError(f"commitment_id not open: {cid_close!r}")
        if str(opened.sender) != str(event.sender):
            raise ValueError(
                f"commitment_id sender mismatch: open.sender={opened.sender!r} close.sender={str(event.sender)!r}"
            )
        dt = float(event.ts) - float(opened.ts)
        if dt < 0.0:
            raise ValueError(f"commitment close ts precedes open ts: dt={dt}")
        self.metrics.commitment_open_to_close_latency_s_sum += dt
        self.metrics.commitment_open_to_close_latency_s_count += 1

        del self._open[cid_close]
        try:
            self._open_order.remove(cid_close)
        except ValueError as e:
            raise RuntimeError("commitment_id missing from open_order") from e
        self.metrics.closes += 1
        return event

    def _close_most_recent_open_for_sender(self, *, sender: str) -> str:
        if not isinstance(sender, str) or not sender.strip():
            raise ValueError("sender must be a non-empty string")
        for cid in reversed(self._open_order):
            opened = self._open.get(cid)
            if opened is not None and str(opened.sender) == sender:
                return str(cid)
        raise ValueError(f"No open commitments for sender={sender!r}")

