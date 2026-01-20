"""Core primitives shared across subsystems.

This package intentionally stays small and dependency-light. It defines
stable contracts (e.g., event envelopes) that higher-level systems can build on.
"""

from core.commitments import CommitmentLedger, CommitmentMetrics
from core.event import EventEnvelope
from core.event_bus import EventBus, EventHandler
from core.event_codec import EventDecoder, EventEncoder
from core.homeostasis import DriveBand, DriveSignal, HomeostaticLoop, IntrinsicDrive

__all__ = [
    "CommitmentLedger",
    "CommitmentMetrics",
    "DriveBand",
    "DriveSignal",
    "EventBus",
    "EventDecoder",
    "EventEncoder",
    "EventEnvelope",
    "EventHandler",
    "HomeostaticLoop",
    "IntrinsicDrive",
]
