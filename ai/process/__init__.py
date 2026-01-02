"""Processes are structured workflows for AI agents."""

from __future__ import annotations

from abc import ABC
from typing import Any
from caramba.console import Logger
from caramba.ai.agent import Agent


class Process(ABC):
    """Base class for AI processes."""
    def __init__(self, agents: dict[str, Agent], name: str):
        self.agents: dict[str, Agent] = agents
        self.name = name
        self.logger = Logger()
        # Shared, process-level transcript.
        # Each item is typically:
        # {"type": "user"|"assistant"|"tool_call"|"tool_result", "author": str, "content": Any}
        self.history: list[dict[str, Any]] = []

    def append_history(self, message: dict[str, Any]) -> None:
        """Append a message to the history."""
        self.history.append(message)

    async def run(self) -> None:
        """Run the process."""
        pass