"""Trace package

Provides deterministic trace capture and replay primitives for Caramba.
This exists so every experiment/process can be replayed byte-for-byte, turning
debugging into a deterministic workflow rather than a probabilistic hunt.
"""

from runtime.trace.reader import TraceReader
from runtime.trace.replay import ReplayRunner
from runtime.trace.schema import TraceEvent
from runtime.trace.writer import TraceWriter

__all__ = [
    "ReplayRunner",
    "TraceEvent",
    "TraceReader",
    "TraceWriter",
]

