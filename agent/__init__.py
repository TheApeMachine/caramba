"""Agent module.

Agents are built using the OpenAI Agent SDK with MCP servers for tools.
"""
from __future__ import annotations

from typing import Any

from agents import Agent, Runner, ModelSettings
from agents.mcp import MCPServer

from agent.context import AgentContext
from agent.tools.paper import PaperTool
from agent.tools.deeplake import DeepLakeTool
from config.persona import PersonaConfig, PersonaType, SharedPersonaConfig
from agent.persona import Persona
from agent.tools import Tool


class Researcher:
    """A researcher with a persona and MCP tool servers."""

    def __init__(
        self,
        persona: PersonaConfig,
    ):
        self.persona = Persona(persona)

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

    async def __aenter__(self, input: str) -> None:
        """Enter async context."""
        result = Runner.run_streamed(
            self.agent,
            input=input
        )

        print("\n")
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.reasoning_text.delta":
                    print(f"\033[33m{event.data.delta}\033[0m", end="", flush=True)
                elif event.data.type == "response.output_text.delta":
                    print(f"\033[32m{event.data.delta}\033[0m", end="", flush=True)

        print("\n")

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        # Cleanup if needed
        pass

    async def run(self, message: str, context: AgentContext | None = None) -> Any:
        """Run the agent with a message.

        Args:
            message: The user message.
            context: Optional context dictionary.

        Returns:
            The agent's response.
        """
        return await Runner.run(
            self.agent,
            input=message,
            context=context or {},
        )
