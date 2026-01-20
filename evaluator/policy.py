"""Policy gate

Enforces hard operational constraints:
- tool capability allowlist/denylist
- tool revision budgets
"""

from __future__ import annotations

from dataclasses import dataclass

from toolchain.events import ToolCapabilities, ToolDefinitionPayload


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Policy configuration."""

    allow_filesystem: bool = True
    allow_network: bool = False
    allow_process: bool = False
    allow_clock: bool = True
    max_tool_revisions: int = 5


@dataclass(slots=True)
class PolicyGate:
    """Policy gate.

    Provides strict enforcement for tool definitions and capabilities.
    """

    config: PolicyConfig

    def validate_tool_definition(self, payload: ToolDefinitionPayload, *, revision_index: int) -> None:
        if not isinstance(payload, ToolDefinitionPayload):
            raise TypeError(f"payload must be ToolDefinitionPayload, got {type(payload).__name__}")
        if int(revision_index) < 0:
            raise ValueError(f"revision_index must be >= 0, got {revision_index}")
        if int(revision_index) >= int(self.config.max_tool_revisions):
            raise RuntimeError(
                f"Tool revision budget exceeded: revision_index={revision_index}, "
                f"max_tool_revisions={int(self.config.max_tool_revisions)}"
            )
        self.validate_capabilities(payload.capabilities)

    def validate_capabilities(self, caps: ToolCapabilities) -> None:
        if not isinstance(caps, ToolCapabilities):
            raise TypeError(f"caps must be ToolCapabilities, got {type(caps).__name__}")
        if bool(caps.filesystem) and not bool(self.config.allow_filesystem):
            raise RuntimeError("Tool capability 'filesystem' is forbidden by policy")
        if bool(caps.network) and not bool(self.config.allow_network):
            raise RuntimeError("Tool capability 'network' is forbidden by policy")
        if bool(caps.process) and not bool(self.config.allow_process):
            raise RuntimeError("Tool capability 'process' is forbidden by policy")
        if bool(caps.clock) and not bool(self.config.allow_clock):
            raise RuntimeError("Tool capability 'clock' is forbidden by policy")

