"""Event handlers for autonomous/self-activation runtime.

This keeps `infer/event_runtime.py` focused on the core prompt→decode loop, while
autonomous behavior (Impulse, Idle consolidation) lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

from core.commitments import CommitmentLedger
from core.event import EventEnvelope
from core.event_bus import EventBus, EventHandler
from infer.event_runtime import CommitmentModeB, EventResponder


class Consolidator(Protocol):
    def consolidate_once(self) -> dict[str, float] | None: ...


@dataclass(slots=True)
class AutonomousModelHandler(EventHandler):
    """EventBus handler that supports Idle/Impulse events.

    - `Idle`: update commitment ledger metrics; optionally run a consolidation step; do NOT emit a model response.
    - `Impulse`: run the model as a self-activation event (same as normal input).
    - other events: normal prompt→respond behavior.
    """

    bus: EventBus
    responder: EventResponder
    ledger: CommitmentLedger = field(default_factory=CommitmentLedger)
    mode_b: CommitmentModeB = field(default_factory=CommitmentModeB)
    consolidator: Consolidator | None = None
    idle_event_type: str = "Idle"
    impulse_event_type: str = "Impulse"

    def __post_init__(self) -> None:
        if not isinstance(self.bus, EventBus):
            raise TypeError(f"bus must be an EventBus, got {type(self.bus).__name__}")
        if not isinstance(self.responder, EventResponder):
            raise TypeError(f"responder must be an EventResponder, got {type(self.responder).__name__}")

    def handle(self, event: EventEnvelope) -> None:
        # Always apply ledger first (tracks open commitments, idle ticks, etc.).
        event = self.ledger.apply(event)

        # Idle: no model response; optional consolidation.
        if str(event.type) == str(self.idle_event_type):
            if self.consolidator is not None:
                _ = self.consolidator.consolidate_once()
            return

        # Default: model responds (Impulse is treated as a normal inbound event).
        resp, aux = self.responder.respond(event)

        # Only inject commitment_delta if commitment head is enabled and aux contains the required key.
        if aux is not None and self.mode_b.key in aux:
            try:
                resp = self.mode_b.inject(resp, aux=aux)
            except KeyError:
                pass

        resp = self.ledger.apply(resp)
        self.bus.publish(resp)


def _tensor_mean(x: Tensor) -> float:
    return float(x.detach().float().mean().item())

