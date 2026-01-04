"""Toolchain package

Provides a safe, test-driven lifecycle for tools:
- tools are versioned artifacts (code + tests + declared capabilities)
- tools execute only via the sandbox runner
- tool acceptance is gated by deterministic unit test results
"""

from caramba.toolchain.events import ToolCapabilities, ToolDefinitionPayload, ToolTestResultPayload
from caramba.toolchain.handler import ToolchainHandler
from caramba.toolchain.registry import ToolRegistry
from caramba.toolchain.sandbox import ToolSandbox
from caramba.toolchain.test_runner import ToolTestRunner

__all__ = [
    "ToolCapabilities",
    "ToolDefinitionPayload",
    "ToolchainHandler",
    "ToolRegistry",
    "ToolSandbox",
    "ToolTestResultPayload",
    "ToolTestRunner",
]

