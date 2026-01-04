"""Tool test runner

Runs a tool's unit tests deterministically inside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass

from caramba.toolchain.events import ToolTestResultPayload
from caramba.toolchain.registry import ToolArtifact
from caramba.toolchain.sandbox import ToolSandbox


@dataclass(slots=True)
class ToolTestRunner:
    """Tool test runner."""

    sandbox: ToolSandbox

    def run_tests(self, artifact: ToolArtifact) -> ToolTestResultPayload:
        """Run unit tests for a tool artifact."""
        if not isinstance(artifact, ToolArtifact):
            raise TypeError(f"artifact must be ToolArtifact, got {type(artifact).__name__}")
        self.sandbox.validate_capabilities(artifact.capabilities)
        code, out = self.sandbox.run_module(
            cwd=artifact.path,
            args=["-m", "unittest", "-q", "tests.py"],
            env={"PYTHONPATH": str(artifact.path)},
        )
        return ToolTestResultPayload(
            name=artifact.name,
            version=artifact.version,
            ok=(int(code) == 0),
            output=out,
        )

