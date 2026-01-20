"""Toolchain handler

Connects the Toolchain (registry + test runner) to the EventBus:
- consumes ToolDefinition events (payload is ToolDefinitionPayload JSON)
- registers the tool version
- runs tests
- publishes ToolTestResult event
"""

from __future__ import annotations

from dataclasses import dataclass

from core.event import EventEnvelope
from core.event_bus import EventBus, EventHandler
from toolchain.events import ToolDefinitionPayload
from toolchain.registry import ToolRegistry
from toolchain.test_runner import ToolTestRunner


@dataclass(slots=True)
class ToolchainHandler(EventHandler):
    """Toolchain EventBus handler.

    This is the control-plane bridge between model-generated tool definitions
    and deterministic tool verification.
    """

    bus: EventBus
    registry: ToolRegistry
    tester: ToolTestRunner
    tool_definition_type: str = "ToolDefinition"
    tool_test_result_type: str = "ToolTestResult"

    def handle(self, event: EventEnvelope) -> None:
        if str(event.type) != str(self.tool_definition_type):
            return
        payload = ToolDefinitionPayload.from_json(event.payload)
        artifact = self.registry.register(payload)
        result = self.tester.run_tests(artifact)
        self.bus.publish(
            EventEnvelope(
                type=str(self.tool_test_result_type),
                payload=result.to_json(),
                sender=str(event.sender),
                priority=int(event.priority),
            )
        )

