"""Prompt building for agents."""
from __future__ import annotations

from agent.context import AgentContext
from config.persona import SharedPersonaConfig


class PromptBuilder:
    """Builds prompts by combining persona, context, and user input."""

    def __init__(self, persona: SharedPersonaConfig):
        """Initialize the prompt builder.

        Args:
            persona: The agent's persona configuration.
        """
        self.persona = persona

    def build(
        self,
        user_input: str,
        context: AgentContext | None = None,
        discussion_history: str | None = None,
    ) -> str:
        """Build a complete prompt.

        Args:
            user_input: The user's message/question.
            context: Optional gathered context.
            discussion_history: Optional prior discussion.

        Returns:
            The formatted prompt.
        """
        parts = []

        # Add context if available
        if context:
            parts.append(context.to_prompt())

        # Add discussion history
        if discussion_history:
            parts.append(f"<discussion_history>\n{discussion_history}\n</discussion_history>")

        # Add user input
        parts.append(user_input)

        return "\n\n".join(parts)


__all__ = ["PromptBuilder"]
