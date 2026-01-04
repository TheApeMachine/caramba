from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caramba.runtime.trace.reader import TraceReader
from caramba.runtime.trace.replay import ReplayRunner
from caramba.runtime.trace.writer import TraceWriter


class TraceTest(unittest.TestCase):
    def test_write_read_replay(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="caramba-trace-"))
        path = tmp / "trace.jsonl"
        w = TraceWriter(path=path)
        w.append(kind="A", payload={"x": 1}, ts=1.0)
        w.append(kind="B", payload={"y": 2}, ts=2.0)

        r = TraceReader(path=path)
        kinds = [e.kind for e in r.events()]
        self.assertEqual(kinds, ["A", "B"])

        seen: list[str] = []
        rr = ReplayRunner(trace_path=path)
        n = rr.run(handler=lambda ev: seen.append(ev.kind))
        self.assertEqual(n, 2)
        self.assertEqual(seen, ["A", "B"])

