"""Trace package

Provides deterministic trace capture and replay primitives for Caramba.
This exists so every experiment/process can be replayed byte-for-byte, turning
debugging into a deterministic workflow rather than a probabilistic hunt.
"""

from caramba.runtime.trace.reader import TraceReader
from caramba.runtime.trace.replay import ReplayRunner
from caramba.runtime.trace.schema import TraceEvent
from caramba.runtime.trace.writer import TraceWriter

__all__ = [
    "ReplayRunner",
    "TraceEvent",
    "TraceReader",
    "TraceWriter",
]

