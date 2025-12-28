"""Agent module.

Agents are built using the OpenAI Agent SDK with MCP servers for tools.
"""
from __future__ import annotations

from typing import Any

from agents import Agent, Runner, ModelSettings
from rich.text import Text

from agent.context import AgentContext
from config.persona import PersonaConfig
from agent.tools import Tool
from console.logger import get_logger


class Researcher:
    """A researcher with a persona and MCP tool servers."""

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
        # If an AgentContext is passed, fold it into the prompt (Runner context expects dict-like).
        ctx: dict[str, Any]
        if isinstance(context, AgentContext):
            message = f"{context.to_prompt()}\n\n{message}"
            ctx = {}
        else:
            ctx = context or {}
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
        ctx: dict[str, Any]
        if isinstance(context, AgentContext):
            message = f"{context.to_prompt()}\n\n{message}"
            ctx = {}
        else:
            ctx = context or {}

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
