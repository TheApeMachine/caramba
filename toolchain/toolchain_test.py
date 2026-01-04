from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus, EventHandler
from caramba.toolchain.events import ToolCapabilities, ToolDefinitionPayload
from caramba.toolchain.handler import ToolchainHandler
from caramba.toolchain.registry import ToolRegistry
from caramba.toolchain.sandbox import ToolSandbox
from caramba.toolchain.test_runner import ToolTestRunner


class ToolchainTest(unittest.TestCase):
    def test_register_and_run_tests(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="caramba-toolchain-"))
        reg = ToolRegistry(root=tmp)
        payload = ToolDefinitionPayload(
            name="decoder",
            version="v1",
            description="d",
            entrypoint="tool:decode",
            code="def decode(x):\n    return x\n",
            tests="import unittest\n\nfrom tool import decode\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(decode(1), 1)\n",
            capabilities=ToolCapabilities(filesystem=False, network=False, process=False, clock=False),
            requirements=[],
        )
        art = reg.register(payload)
        runner = ToolTestRunner(sandbox=ToolSandbox())
        res = runner.run_tests(art)
        self.assertTrue(res.ok)

    def test_event_bus_flow(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="caramba-toolchain-bus-"))
        reg = ToolRegistry(root=tmp)
        runner = ToolTestRunner(sandbox=ToolSandbox())
        bus = EventBus()
        handler = ToolchainHandler(bus=bus, registry=reg, tester=runner)
        rec = Recorder()
        bus.subscribe("ToolDefinition", handler)
        bus.subscribe("ToolTestResult", rec)

        payload = ToolDefinitionPayload(
            name="decoder2",
            version="v1",
            description="d",
            entrypoint="tool:decode",
            code="def decode(x):\n    return x\n",
            tests="import unittest\n\nfrom tool import decode\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(decode(1), 1)\n",
            capabilities=ToolCapabilities(),
            requirements=[],
        )
        bus.publish(EventEnvelope(type="ToolDefinition", payload=payload.to_json(), sender="agent"))
        _ = bus.drain()
        self.assertTrue(rec.events)
        self.assertEqual(rec.events[-1].type, "ToolTestResult")


class Recorder(EventHandler):
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    def handle(self, event: EventEnvelope) -> None:
        self.events.append(event)

