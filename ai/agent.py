"""Agent wrapper around Google ADK.

Provides a unified interface for creating agents that can participate
in the A2A protocol and use MCP tools.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from google.adk import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.genai import types as genai_types

from .persona import PersonaLoader, persona_to_agent_card
from .tools import (
    BestEffortMcpToolset,
    connection_params_for,
    endpoint_is_healthy,
    iter_persona_tool_names,
    load_mcp_endpoints,
)
from .types import PersonaConfig

_logger = logging.getLogger(__name__)


class Agent:
    """Wrapper around Google ADK Agent with persona configuration.

    Simplifies agent creation by loading configuration from persona files
    and managing MCP tool integration.
    """

    def __init__(
        self,
        persona: str | PersonaConfig,
        tools: list[Callable[..., Any]] | None = None,
        sub_agents: list[Agent] | None = None,
        loader: PersonaLoader | None = None,
    ) -> None:
        """Initialize an agent from a persona.

        Args:
            persona: Persona type string or PersonaConfig instance.
            tools: Optional list of tool functions to add.
            sub_agents: Optional list of sub-agents to delegate to.
            loader: PersonaLoader instance. Creates one if not provided.
        """
        self.loader = loader or PersonaLoader()

        if isinstance(persona, str):
            self.config = self.loader.load(persona)
        else:
            self.config = persona

        self._tools = tools or []
        self._sub_agents = sub_agents or []
        self._adk_agent: ADKAgent | None = None

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the agent description."""
        return self.config.description

    def get_agent_card(self, base_url: str):
        """Get an A2A AgentCard for this agent.

        Args:
            base_url: The base URL where this agent is hosted.

        Returns:
            An A2A AgentCard.
        """
        return persona_to_agent_card(self.config, base_url)

    def add_tool(self, tool: Callable[..., Any]) -> None:
        """Add a tool function to this agent.

        Args:
            tool: A callable that will be exposed as a tool.
        """
        self._tools.append(tool)
        self._adk_agent = None  # Invalidate cached agent

    def add_sub_agent(self, agent: Agent) -> None:
        """Add a sub-agent for delegation.

        Args:
            agent: The sub-agent to add.
        """
        self._sub_agents.append(agent)
        self._adk_agent = None  # Invalidate cached agent

    def build(self) -> ADKAgent:
        """Build and return the underlying ADK Agent.

        Returns:
            The configured ADK Agent instance.
        """
        if self._adk_agent is not None:
            return self._adk_agent

        # Enable LiteLLM debug mode
        import litellm

        # Use model from config (should already include provider prefix) or env override
        model_name = os.getenv("LITELLM_MODEL", self.config.model)
        model = LiteLlm(model=model_name)

        # Configure for streaming responses
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.config.temperature,
        )

        # Build tools list from callables
        adk_tools: list[Any] = []
        for tool in self._tools:
            adk_tools.append(FunctionTool(tool))

        # Load MCP toolsets from persona config
        tool_names = iter_persona_tool_names(self.config.tools)
        if tool_names:
            endpoints = load_mcp_endpoints()
            for tool_name in tool_names:
                if tool_name not in endpoints:
                    _logger.warning(f"MCP tool '{tool_name}' not found in endpoints")
                    continue
                endpoint = endpoints[tool_name]
                if not endpoint_is_healthy(endpoint):
                    _logger.warning(f"MCP tool '{tool_name}' is not healthy, skipping")
                    continue
                params = connection_params_for(endpoint)
                toolset = BestEffortMcpToolset(
                    connection_params=params,
                    label=tool_name,
                )
                adk_tools.append(toolset)
                _logger.info(f"Loaded MCP toolset: {tool_name}")

        adk_sub_agents = []
        for sub_agent in self._sub_agents:
            adk_sub_agents.append(sub_agent.build())

        # Only pass non-empty lists to avoid type issues
        kwargs: dict[str, Any] = {
            "name": self.config.name.lower().replace(" ", "_"),
            "model": model,
            "description": self.config.description,
            "instruction": self._build_instruction,
            "generate_content_config": generate_config,
        }
        if adk_tools:
            kwargs["tools"] = adk_tools
        if adk_sub_agents:
            kwargs["sub_agents"] = adk_sub_agents

        self._adk_agent = ADKAgent(**kwargs)
        return self._adk_agent

    def _build_instruction(self, context: ReadonlyContext) -> str:
        """Build the instruction string for the agent.

        Args:
            context: The readonly context from ADK.

        Returns:
            The instruction string.
        """
        base_instruction = self.config.instructions

        # Add sub-agent information if available
        if self._sub_agents:
            sub_agent_info = "\n\nAvailable sub-agents:\n"
            for sub in self._sub_agents:
                sub_agent_info += f"- {sub.name}: {sub.description}\n"
            return base_instruction + sub_agent_info

        return base_instruction


class AgentFactory:
    """Factory for creating agents from persona configurations.

    Provides a centralized way to create agents with proper
    tool and sub-agent configuration.
    """

    def __init__(
        self,
        personas_dir: str | None = None,
        tools_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            personas_dir: Path to personas directory.
            tools_registry: Dictionary mapping tool names to callables.
        """
        self.loader = PersonaLoader(personas_dir)
        self.tools_registry = tools_registry or {}

    def register_tool(self, name: str, tool: Callable[..., Any]) -> None:
        """Register a tool function.

        Args:
            name: The tool name (must match persona YAML).
            tool: The tool callable.
        """
        self.tools_registry[name] = tool

    def create(
        self,
        persona_type: str,
        sub_agents: list[Agent] | None = None,
    ) -> Agent:
        """Create an agent from a persona type.

        Args:
            persona_type: The persona type to load.
            sub_agents: Optional sub-agents to add.

        Returns:
            The configured Agent instance.
        """
        config = self.loader.load(persona_type)

        # Resolve tools from registry
        tools = []
        for tool_name in config.tools:
            if tool_name in self.tools_registry:
                tools.append(self.tools_registry[tool_name])

        return Agent(
            persona=config,
            tools=tools,
            sub_agents=sub_agents,
            loader=self.loader,
        )
