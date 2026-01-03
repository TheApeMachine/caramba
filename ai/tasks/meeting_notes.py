"""Meeting notes extraction task.

This task extracts ideas and consensus levels from conversation history
and saves them as markdown formatted meeting notes.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.ai.tasks.task import Task
from caramba.console import logger


class MeetingNotesTask(Task):
    """Task that extracts meeting notes (ideas and consensus) from conversation history."""

    def __init__(
        self,
        conversation_history: list[dict[str, Any]],
        persona_path: Path | None = None,
        prompt_path: Path | None = None,
        output_dir: Path | None = None,
    ):
        super().__init__("meeting_notes")
        self.conversation_history = conversation_history
        self.persona_path = persona_path or Path("config/personas/note_taker.yml")
        self.prompt_path = prompt_path or Path("config/prompts/meeting_notes.yml")
        self.output_dir = output_dir or Path("artifacts/ai/meeting_notes")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _filter_history(self) -> list[dict[str, Any]]:
        """Filter out tool calls and tool results from history."""
        filtered: list[dict[str, Any]] = []
        for msg in self.conversation_history:
            mtype = str(msg.get("type", ""))
            # Skip tool calls and tool results
            if mtype in ("tool_call", "tool_result"):
                continue
            filtered.append(msg)
        return filtered

    def _render_history(self) -> str:
        """Render filtered conversation history as markdown (excluding tool calls/results)."""
        filtered = self._filter_history()
        lines: list[str] = []
        for msg in filtered:
            mtype = str(msg.get("type", ""))
            author = str(msg.get("author", ""))
            content = msg.get("content", "")
            if mtype == "user":
                lines.append(f"- **{author}**: {content}")
            elif mtype == "assistant":
                lines.append(f"- **{author}**: {content}")
            else:
                lines.append(f"- **{author}** ({mtype}): {content}")
        return "\n".join(lines).strip()

    def _load_prompt(self) -> str:
        """Load the extraction prompt from YAML config."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        with open(self.prompt_path, "r") as f:
            data = yaml.safe_load(f) or {}
            meeting_notes_config = data.get("meeting_notes", {})
            prompt_template = meeting_notes_config.get("extraction_prompt", "")
            if not prompt_template:
                raise ValueError(f"No extraction_prompt found in {self.prompt_path}")
            return prompt_template

    async def _extract_meeting_notes(self) -> str | None:
        """Extract meeting notes from conversation history using an AI agent."""
        filtered_history = self._filter_history()
        if not filtered_history:
            return None

        # Load persona from YAML
        if not self.persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_path}")

        persona = Persona.from_yaml(self.persona_path)

        # Create Agent wrapper (handles model, session, runner)
        agent = Agent(persona=persona, app_name="meeting_notes", user_id="system")

        # Render filtered history for extraction
        conversation_text = self._render_history()

        # Load prompt template and format it
        prompt_template = self._load_prompt()
        extraction_prompt = prompt_template.format(conversation_history=conversation_text)

        try:
            # Run the agent asynchronously
            return await agent.run_async(extraction_prompt)

        except Exception as e:
            logger.warning(f"Meeting notes extraction failed: {e}")
            return None

    def _save_meeting_notes(self, notes: str) -> Path:
        """Save meeting notes to a timestamped markdown file."""
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_notes_{timestamp}.md"
        filepath = self.output_dir / filename

        # Add a header if the agent didn't provide one
        if not notes.strip().startswith("#"):
            header_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes = f"# Meeting Notes\n**Generated:** {header_ts}\n\n{notes}"

        # Write to file
        filepath.write_text(notes, encoding="utf-8")

        return filepath

    async def run_async(self) -> dict[str, Any]:
        """Run the meeting notes extraction task asynchronously."""
        logger.info("Extracting meeting notes from conversation history...")

        # Extract meeting notes as markdown text
        notes = await self._extract_meeting_notes()

        if notes:
            logger.info("Saving meeting notes...")
            filepath = self._save_meeting_notes(notes)
            logger.success(f"Meeting notes saved to: {filepath}")

            # Count distinct ideas roughly by markdown headers
            num_ideas = notes.count("## ") + notes.count("### ")

            return {
                "success": True,
                "notes_extracted": True,
                "filepath": str(filepath),
                "num_ideas": num_ideas,
            }
        else:
            logger.info("No meeting notes extracted (empty history or extraction failed)")
            return {
                "success": True,
                "notes_extracted": False,
            }
