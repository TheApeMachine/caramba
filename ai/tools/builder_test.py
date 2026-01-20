from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from .builder import ToolBuilderHandler
from core.event import EventEnvelope
from core.event_bus import EventBus, EventHandler


class _Recorder(EventHandler):
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    def handle(self, event: EventEnvelope) -> None:
        self.events.append(event)


class ToolBuilderTest(unittest.TestCase):
    def test_tool_definition_scaffolds_files_and_emits_registered(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="caramba-toolbuilder-"))
        try:
            bus = EventBus()
            rec = _Recorder()
            bus.subscribe("ToolRegistered", rec)
            bus.subscribe("ToolRejected", rec)
            bus.subscribe("ToolDefinition", ToolBuilderHandler(bus=bus, out_dir=tmp))

            bus.publish(
                EventEnvelope(
                    type="ToolDefinition",
                    payload={
                        "name": "my_tool",
                        "description": "desc",
                        "implementation": "def run():\n    return 1\n",
                        "requirements": ["requests==2.31.0"],
                    },
                    sender="agent",
                )
            )
            _ = bus.drain()
            self.assertTrue(rec.events)
            self.assertEqual(rec.events[-1].type, "ToolRegistered")
            self.assertTrue((tmp / "my_tool" / "tool.py").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

