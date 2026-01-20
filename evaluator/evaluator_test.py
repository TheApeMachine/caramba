from __future__ import annotations

import unittest

from core.event import EventEnvelope
from evaluator.policy import PolicyConfig, PolicyGate
from evaluator.validity import ValidityGate
from toolchain.events import ToolCapabilities, ToolDefinitionPayload


class EvaluatorTest(unittest.TestCase):
    def test_validity_gate(self) -> None:
        g = ValidityGate()
        ev = EventEnvelope(type="X", payload={"a": 1}, sender="s")
        g.validate_event(ev)

    def test_policy_gate_rejects_network(self) -> None:
        g = PolicyGate(config=PolicyConfig(allow_network=False))
        p = ToolDefinitionPayload(
            name="t",
            version="v1",
            description="d",
            entrypoint="tool:run",
            code="def run():\n    return 1\n",
            tests="import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            capabilities=ToolCapabilities(network=True),
            requirements=[],
        )
        with self.assertRaises(RuntimeError):
            g.validate_tool_definition(p, revision_index=0)

