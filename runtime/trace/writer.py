"""Trace writer

Writes newline-delimited JSON trace events to disk. This is used to guarantee
that every run can be replayed deterministically.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.trace.schema import TraceEvent


@dataclass(slots=True)
class TraceWriter:
    """Trace writer.

    Used to append structured events into a JSONL trace file.
    """

    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        if not str(self.path):
            raise ValueError("TraceWriter.path must be non-empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, kind: str, payload: dict[str, Any], ts: float | None = None) -> None:
        """Append one trace event."""
        if ts is None:
            ts = time.time()
        ev = TraceEvent(kind=str(kind), payload=dict(payload), ts=float(ts))
        line = json.dumps(ev.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

