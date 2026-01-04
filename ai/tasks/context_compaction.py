"""Context compaction task.

This task summarizes older conversation history into a compact "memory" message.
It is used when the shared transcript grows too large for the smallest provider
context window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.ai.tasks import Task


class ContextCompactionTask(Task):
    """Context compaction task.

    Used to preserve intent while reducing token footprint by replacing an older
    transcript segment with a concise summary that keeps:
    - key decisions
    - hypotheses
    - references to artifacts/paths
    - open questions and next actions
    """

    def __init__(self) -> None:
        super().__init__("context_compaction")
        self.agent = Agent(
            persona=Persona.from_yaml(Path("config/personas/context_compactor.yml")),
            app_name="context_compaction",
            user_id="system",
        )

    async def run_async(self) -> dict[str, Any]:
        """Summarize the task history into a compact memory string."""
        if not self.history:
            raise ValueError("ContextCompactionTask requires non-empty history")

        memory = await self.agent.run_async(
            types.Content(
                role="user",
                parts=[part for item in self.history if item.parts for part in item.parts],
            )
        )
        if not isinstance(memory, str) or not memory.strip():
            raise RuntimeError("Context compaction produced empty output")

        return {"memory": memory}

