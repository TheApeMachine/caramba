"""Agent module.

Agents are built using the OpenAI Agent SDK with MCP servers for tools.
"""
from __future__ import annotations

from typing import Any

from agents import Agent, Runner, ModelSettings
from rich.text import Text

from caramba.agent.context import AgentContext
from caramba.config.persona import PersonaConfig
from caramba.agent.tools import Tool
from caramba.console.logger import get_logger


class Researcher:
    """A researcher with a persona and MCP tool servers."""

    @staticmethod
    def _prepare_message_and_ctx(
        context: dict[str, Any] | AgentContext | None,
        message: str,
    ) -> tuple[str, dict[str, Any]]:
        """Normalize Runner context and optionally fold AgentContext into the prompt.

        Runner expects a dict-like context. When an AgentContext is provided, we fold
        its prompt representation into the message and pass an empty dict context.
        """
        if isinstance(context, AgentContext):
            return f"{context.to_prompt()}\n\n{message}", {}
        return message, (context or {})

    def __init__(
        self,
        persona: PersonaConfig,
    ):
        self.persona = persona
        self.logger = get_logger()

        self.agent = Agent(
            name=persona.name,
            instructions=persona.instructions,
            model=persona.model,
            model_settings=ModelSettings(
                temperature=persona.temperature,
            ),
            mcp_servers=[
                Tool(name).mcp_client() for name in persona.mcp_servers
            ]
        )

    async def __aenter__(self) -> "Researcher":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        # Cleanup if needed
        pass

    async def run(self, message: str, context: dict[str, Any] | AgentContext | None = None) -> Any:
        """Run the agent with a message.

        Args:
            message: The user message.
            context: Optional context dictionary.

        Returns:
            The agent's response.
        """
        message, ctx = self._prepare_message_and_ctx(context, message)
        return await Runner.run(
            self.agent,
            input=message,
            context=ctx,
        )

    async def run_streamed_to_console(
        self,
        message: str,
        context: dict[str, Any] | AgentContext | None = None,
        *,
        show_reasoning: bool = True,
        show_output: bool = True,
    ) -> Any:
        """Run the agent and stream deltas to the rich console logger."""
        message, ctx = self._prepare_message_and_ctx(context, message)

        result = Runner.run_streamed(self.agent, input=message, context=ctx)

        # Rich console output (no raw ANSI).
        self.logger.console.print()
        async for event in result.stream_events():
            if event.type != "raw_response_event":
                continue
            if event.data.type == "response.reasoning_text.delta" and show_reasoning:
                self.logger.console.print(Text(str(event.data.delta), style="warning"), end="")
            elif event.data.type == "response.output_text.delta" and show_output:
                self.logger.console.print(Text(str(event.data.delta), style="success"), end="")
        self.logger.console.print()

        return result
