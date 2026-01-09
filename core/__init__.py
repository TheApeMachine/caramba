"""Core primitives shared across subsystems.

This package intentionally stays small and dependency-light. It defines
stable contracts (e.g., event envelopes) that higher-level systems can build on.
"""

from caramba.core.commitments import CommitmentLedger, CommitmentMetrics
from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus, EventHandler
from caramba.core.event_codec import JsonEventDecoder, JsonEventEncoder
from caramba.core.homeostasis import DriveBand, DriveSignal, HomeostaticLoop, IntrinsicDrive

__all__ = [
    "CommitmentLedger",
    "CommitmentMetrics",
    "DriveBand",
    "DriveSignal",
    "EventBus",
    "EventEnvelope",
    "EventHandler",
    "HomeostaticLoop",
    "IntrinsicDrive",
    "JsonEventDecoder",
    "JsonEventEncoder",
]
