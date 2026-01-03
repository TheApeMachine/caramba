"""Base task class for all tasks."""

from __future__ import annotations

from abc import ABC


class Task(ABC):
    """Task is a base class for all tasks."""
    def __init__(self, name: str):
        self.name = name

    def run(self) -> None:
        """Run the task."""
        pass
