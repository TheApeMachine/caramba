"""Event bus for the peer-to-peer "organism" architecture.

This is a small, in-memory dispatcher for EventEnvelope instances.
It is intentionally simple and strict:
- publishing an event with no subscribers raises an error (no silent drops)
- handlers are objects (no loose functions), matching the framework style
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from caramba.core.event import EventEnvelope


class EventHandler(ABC):
    @abstractmethod
    def handle(self, event: EventEnvelope) -> None: ...


@dataclass(slots=True)
class EventBus:
    """Priority-ordered event dispatcher."""

    _seq: int = 0
    _queue: list[tuple[int, float, int, EventEnvelope]] = field(default_factory=list)
    _subs: defaultdict[str, list[EventHandler]] = field(default_factory=lambda: defaultdict(list))

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        et = str(event_type)
        if not et:
            raise ValueError("event_type must be non-empty")
        if not isinstance(handler, EventHandler):
            raise TypeError(f"handler must be an EventHandler, got {type(handler).__name__}")
        hs = self._subs[et]
        if handler in hs:
            raise ValueError(f"Handler already subscribed for event_type={et!r}")
        hs.append(handler)

    def publish(self, event: EventEnvelope) -> None:
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"event must be an EventEnvelope, got {type(event).__name__}")
        et = str(event.type)
        hs = (self._subs.get(et) or []) + (self._subs.get("*") or [])
        if not hs:
            raise RuntimeError(f"No subscribers for event type {et!r}")
        # Use a max-priority heap (invert priority).
        pri = -int(event.priority)
        ts = float(event.ts)
        self._seq += 1
        heapq.heappush(self._queue, (pri, ts, int(self._seq), event))

    def pending(self) -> int:
        return len(self._queue)

    def dispatch_one(self) -> EventEnvelope:
        if not self._queue:
            raise RuntimeError("No pending events to dispatch")
        _pri, _ts, _seq, ev = heapq.heappop(self._queue)
        hs = (self._subs.get(str(ev.type)) or []) + (self._subs.get("*") or [])
        if not hs:
            raise RuntimeError(f"No subscribers for event type {str(ev.type)!r}")
        for h in hs:
            h.handle(ev)
        return ev

    def drain(self, *, max_events: int | None = None) -> int:
        n_max = int(max_events) if max_events is not None else None
        if n_max is not None and n_max < 1:
            raise ValueError(f"max_events must be >= 1, got {n_max}")
        n = 0
        while self._queue and (n_max is None or n < n_max):
            self.dispatch_one()
            n += 1
        return int(n)

