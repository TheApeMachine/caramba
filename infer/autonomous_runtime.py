"""Autonomous runtime loop (idle compute + homeostatic impulses).

This module wraps the synchronous EventBus with an async loop that:
- drains inbound events when present
- otherwise evaluates a HomeostaticLoop and publishes an Impulse event when urgency exceeds threshold
- otherwise publishes an Idle tick event (so subsystems like CommitmentLedger can update metrics)

This is intentionally lightweight: it does not start servers, threads, or background workers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus
from caramba.core.homeostasis import HomeostaticLoop

MetricsFn = Callable[[], Mapping[str, float]]
ConsolidateFn = Callable[[], dict[str, float] | None]


def _default_metrics() -> Mapping[str, float]:
    return {}


@dataclass(slots=True)
class AutonomousRuntime:
    """Drive the EventBus when external inputs are absent."""

    bus: EventBus
    homeostasis: HomeostaticLoop
    sender: str = "runtime"
    poll_interval_ms: int = 50
    idle_event_type: str = "Idle"
    # Optional callback to provide metrics used by homeostasis.
    metrics_fn: MetricsFn = field(default_factory=lambda: _default_metrics)
    # Optional idle-time consolidation hook.
    consolidate_fn: ConsolidateFn | None = None
    # Max events to dispatch per tick to avoid starvation.
    dispatch_budget: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.bus, EventBus):
            raise TypeError(f"bus must be an EventBus, got {type(self.bus).__name__}")
        if not isinstance(self.homeostasis, HomeostaticLoop):
            raise TypeError(f"homeostasis must be a HomeostaticLoop, got {type(self.homeostasis).__name__}")
        if not isinstance(self.sender, str) or not self.sender.strip():
            raise ValueError("sender must be a non-empty string")
        self.poll_interval_ms = int(self.poll_interval_ms)
        if self.poll_interval_ms < 1:
            raise ValueError(f"poll_interval_ms must be >= 1, got {self.poll_interval_ms}")
        self.dispatch_budget = int(self.dispatch_budget)
        if self.dispatch_budget < 1:
            raise ValueError(f"dispatch_budget must be >= 1, got {self.dispatch_budget}")

    def tick_once(self) -> int:
        """Run one scheduling tick. Returns number of events dispatched."""
        # Drain pending work first.
        if self.bus.pending() > 0:
            return int(self.bus.drain(max_events=int(self.dispatch_budget)))

        # No external events pending -> evaluate homeostasis.
        metrics = self.metrics_fn()
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics_fn must return a Mapping[str,float], got {type(metrics).__name__}")
        metrics_f = {str(k): float(v) for k, v in metrics.items()}

        impulse = self.homeostasis.impulse(metrics_f)
        if impulse is not None:
            # HomeostaticLoop already sets type="Impulse" and embeds signals/metrics.
            self.bus.publish(impulse)
            return int(self.bus.drain(max_events=int(self.dispatch_budget)))

        # True idle: optional consolidation, then publish an Idle tick for subsystems that track time/budgets.
        if self.consolidate_fn is not None:
            _ = self.consolidate_fn()
        self.bus.publish(
            EventEnvelope(
                type=str(self.idle_event_type),
                payload={"metrics": metrics_f, "ts": float(time.time())},
                sender=str(self.sender),
                priority=0,
            )
        )
        return int(self.bus.drain(max_events=int(self.dispatch_budget)))

    async def run(self) -> None:
        """Run forever."""
        while True:
            _ = self.tick_once()
            await asyncio.sleep(float(self.poll_interval_ms) / 1000.0)

