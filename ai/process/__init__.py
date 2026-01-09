"""Processes are structured workflows for AI agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from caramba.console import Logger
from google.genai import types

if TYPE_CHECKING:
    from caramba.ai.agent import Agent


class Process(ABC):
    """Base class for AI processes."""

    # Maximum number of history messages to retain (prevents unbounded memory growth).
    MAX_HISTORY_SIZE: int = 10000

    def __init__(self, agents: dict[str, Agent], name: str):
        self.agents: dict[str, Agent] = agents
        self.name = name
        self.logger = Logger()
        self.history: list[types.Content] = []

    def append_history(self, message: types.Content) -> None:
        """Append a message to the history with LRU eviction."""
        self.history.append(message)
        # Evict oldest entries if history exceeds max size.
        if len(self.history) > self.MAX_HISTORY_SIZE:
            self.history = self.history[-self.MAX_HISTORY_SIZE:]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()

    @abstractmethod
    async def run(self) -> None:
        """Run the process."""
        pass