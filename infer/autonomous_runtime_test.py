from __future__ import annotations

import unittest

from core.event import EventEnvelope
from core.event_bus import EventBus, EventHandler
from core.homeostasis import DriveBand, HomeostaticLoop, IntrinsicDrive
from infer.autonomous_runtime import AutonomousRuntime


class _Recorder(EventHandler):
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    def handle(self, event: EventEnvelope) -> None:
        self.events.append(event)


class AutonomousRuntimeTest(unittest.TestCase):
    def test_emits_idle_when_no_impulse(self) -> None:
        bus = EventBus()
        rec = _Recorder()
        bus.subscribe("Idle", rec)

        homeo = HomeostaticLoop(
            drives=(IntrinsicDrive(name="x", metric="x", band=DriveBand(0.0, 1.0)),),
            sender="homeo",
            impulse_threshold=10.0,  # unreachable
        )

        rt = AutonomousRuntime(bus=bus, homeostasis=homeo, sender="rt", metrics_fn=lambda: {"x": 0.5})
        n = rt.tick_once()
        self.assertGreaterEqual(n, 1)
        self.assertTrue(rec.events)
        self.assertEqual(rec.events[-1].type, "Idle")

    def test_emits_impulse_when_threshold_exceeded(self) -> None:
        bus = EventBus()
        rec = _Recorder()
        bus.subscribe("Impulse", rec)

        homeo = HomeostaticLoop(
            drives=(IntrinsicDrive(name="x", metric="x", band=DriveBand(0.0, 1.0), weight=1.0),),
            sender="homeo",
            impulse_threshold=0.0,  # any deviation triggers
        )

        # x=2 deviates by 1.0
        rt = AutonomousRuntime(bus=bus, homeostasis=homeo, sender="rt", metrics_fn=lambda: {"x": 2.0})
        n = rt.tick_once()
        self.assertGreaterEqual(n, 1)
        self.assertTrue(rec.events)
        self.assertEqual(rec.events[-1].type, "Impulse")

