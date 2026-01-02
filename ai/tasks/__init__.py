"""Tasks that can be attached to processes

These are general house-keeping tasks that can be attached to processes.
"""

from __future__ import annotations

from abc import ABC

from caramba.ai.tasks.knowledge import KnowledgeExtractionTask

__all__ = ["Task", "KnowledgeExtractionTask"]


class Task(ABC):
    """Task is a base class for all tasks."""
    def __init__(self, name: str):
        self.name = name

    def run(self) -> None:
        """Run the task."""
        pass