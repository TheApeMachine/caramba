"""Replay runner

Replays a trace file deterministically by emitting TraceEvent records in order.
Higher-level systems can wire these records back into their event buses and tool
executors to reproduce behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from runtime.trace.reader import TraceReader
from runtime.trace.schema import TraceEvent


@dataclass(slots=True)
class ReplayRunner:
    """Replay runner.

    Provides a strict mechanism to replay a trace by calling a handler for each event.
    """

    trace_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.trace_path, Path):
            self.trace_path = Path(self.trace_path)

    def run(self, *, handler: Callable[[TraceEvent], None]) -> int:
        """Replay all events, returning count."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        reader = TraceReader(path=self.trace_path)
        n = 0
        for ev in reader.events():
            handler(ev)
            n += 1
        return int(n)

