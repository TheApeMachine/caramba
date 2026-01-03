"""Meeting notes extraction task.

This task extracts ideas and consensus levels from conversation history
and saves them as markdown formatted meeting notes.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from google.genai import types

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.ai.tasks import Task
from caramba.console import logger


class MeetingNotesTask(Task):
    """Task that extracts meeting notes (ideas and consensus) from conversation history."""
    def __init__(
        self,
    ):
        super().__init__("meeting_notes")
        self.agent = Agent(
            persona=Persona.from_yaml(Path("config/personas/note_taker.yml")),
            app_name="meeting_notes",
            user_id="system",
        )

    def run(self, history: list[types.Content]) -> dict[str, Any]:
        """Run the task synchronously (wrapper around `run_async()`)."""
        return super().run(self.filter_history(history))

    def filter_history(self, history: list[types.Content]) -> list[types.Content]:
        """Filter out tool calls and tool results from history."""
        return [msg for msg in history if msg.role not in ("tool_call", "tool_result")]

    async def run_async(self) -> None:
        """Run the meeting notes extraction task asynchronously."""
        logger.info("Extracting meeting notes from conversation history...")

        if not self.history:
            logger.warning("No conversation history to extract meeting notes from.")
            return

        try:
            notes = await self.agent.run_async(
                types.Content(
                    role="user",
                    parts=[
                        part for item in self.history if item.parts for part in item.parts
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Error extracting meeting notes: {e}")
            return

        logger.info(f"Meeting notes extracted: {notes}")

