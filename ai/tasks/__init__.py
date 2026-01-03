"""Tasks that can be attached to processes

These are general house-keeping tasks that can be attached to processes.
"""
from __future__ import annotations
from typing import Any
import asyncio
from abc import ABC, abstractmethod

from google.genai import types


class Task(ABC):
    """Task is a base class for all tasks."""
    def __init__(self, name: str):
        self.name = name
        self.history: list[types.Content] = []

    @abstractmethod
    async def run_async(self) -> dict[str, Any]:
        """Run the task asynchronously and return a structured result."""
        raise NotImplementedError

    def run(self, history: list[types.Content]) -> dict[str, Any]:
        """Run the task synchronously (wrapper around `run_async()`)."""
        self.history = history

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async())
        raise RuntimeError(
            "Task.run() cannot be called from an active event loop; use `await Task.run_async(...)`."
        )