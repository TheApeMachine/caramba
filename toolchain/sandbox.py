"""Tool sandbox

Executes tools and tests in a separate Python process with strict limits.
This is the enforcement point for capabilities and resource budgets.
"""

from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.toolchain.events import ToolCapabilities


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Sandbox resource limits."""

    timeout_sec: float = 5.0


@dataclass(slots=True)
class ToolSandbox:
    """Tool sandbox runner.

    Uses subprocess execution. Capabilities are enforced by configuration:
    - network/process/filesystem permissions must be implemented via OS/container policy
    - this runner is strict about what it allows and raises if unsupported
    """

    python: str = sys.executable
    limits: SandboxLimits = SandboxLimits()

    def validate_capabilities(self, caps: ToolCapabilities) -> None:
        """Validate declared capabilities against available sandbox enforcement."""
        if caps.network:
            raise RuntimeError("Network capability is not supported by ToolSandbox; use a containerized runner.")
        if caps.process:
            raise RuntimeError("Process capability is not supported by ToolSandbox; use a containerized runner.")
        # filesystem/clock are allowed in this simple runner by default.

    def run_module(self, *, cwd: Path, args: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
        """Run a Python module in a subprocess."""
        if not isinstance(cwd, Path):
            raise TypeError("cwd must be a Path")
        if not cwd.exists():
            raise FileNotFoundError(f"Sandbox cwd not found: {cwd}")
        p = subprocess.run(
            [self.python, *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=float(self.limits.timeout_sec),
            check=False,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return int(p.returncode), str(out)

