"""Trace reader

Reads newline-delimited JSON traces and yields TraceEvent records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from runtime.trace.schema import TraceEvent


@dataclass(slots=True)
class TraceReader:
    """Trace reader.

    Used to load and iterate over a JSONL trace file.
    """

    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.path}")

    def events(self) -> Iterator[TraceEvent]:
        """Iterate events in file order."""
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise TypeError(f"Trace line must decode to dict, got {type(obj).__name__}")
                yield TraceEvent.from_json(obj)

