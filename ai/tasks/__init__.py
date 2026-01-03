"""Tasks that can be attached to processes

These are general house-keeping tasks that can be attached to processes.
"""

from __future__ import annotations

from caramba.ai.tasks.task import Task
from caramba.ai.tasks.knowledge import KnowledgeExtractionTask
from caramba.ai.tasks.meeting_notes import MeetingNotesTask

__all__ = ["Task", "KnowledgeExtractionTask", "MeetingNotesTask"]