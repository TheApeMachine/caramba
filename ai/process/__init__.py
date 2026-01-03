"""Processes are structured workflows for AI agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from caramba.console import Logger
from caramba.ai.agent import Agent
from google.genai import types


class Process(ABC):
    """Base class for AI processes."""
    def __init__(self, agents: dict[str, Agent], name: str):
        self.agents: dict[str, Agent] = agents
        self.name = name
        self.logger = Logger()
        self.history: list[types.Content] = []

    def append_history(self, message: types.Content) -> None:
        """Append a message to the history."""
        self.history.append(message)

    @abstractmethod
    async def run(self) -> None:
        """Run the process."""
        pass