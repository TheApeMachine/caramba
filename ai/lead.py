"""Team lead agent that delegates to team members.

Team leads are responsible for coordinating work within their team,
delegating to specialized members, and synthesizing results.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from google.adk import Agent as ADKAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from a2a.types import (
    Message,
    Part,
    Task,
    TaskState,
)

from .connection import ConnectionManager, TaskCallback
from .persona import PersonaLoader, persona_to_agent_card
from .retry import http_get_with_retry
from .team import TeamLoader
from .tools import (
    BestEffortMcpToolset,
    connection_params_for,
    endpoint_is_healthy,
    iter_persona_tool_names,
    load_mcp_endpoints,
)

_logger = logging.getLogger(__name__)


class LeadAgent:
    """A team lead agent that can delegate to team members.

    Team leads coordinate work within their team by delegating
    to specialized members and synthesizing their outputs.
    """

    def __init__(
        self,
        persona_type: str,
        httpx_client: httpx.AsyncClient,
        team_config: str = "default",
        webhook_base_url: str | None = None,
    ) -> None:
        """Initialize the team lead.

        Args:
            persona_type: The persona type for this lead.
            httpx_client: The httpx client for A2A connections.
            team_config: The team configuration file name.
            webhook_base_url: Base URL for this agent's webhook (for push notifications).
        """
        self.persona_type = persona_type
        self.httpx_client = httpx_client
        self.team_config = team_config
        self.webhook_base_url = webhook_base_url

        self.persona_loader = PersonaLoader()
        self.team_loader = TeamLoader()
        self.connections = ConnectionManager(httpx_client)

        # Load persona
        self.config = self.persona_loader.load(persona_type)
        self._members_info: str = ""
        self._adk_agent: ADKAgent | None = None

        # Track delegated tasks for async updates
        self._delegated_tasks: dict[str, dict[str, Any]] = {}

        # Get team members
        self.members = self.team_loader.get_members_for_lead(
            persona_type, team_config
        )

        # Initialize connections in background
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._init_member_connections())

    async def _on_task_update(self, task: Task) -> None:
        """Handle updates for delegated tasks."""
        _logger.info(f"LeadAgent received update for task {task.id}: {task.status.state}")
        
        # Update local tracking
        if task.id in self._delegated_tasks:
            self._delegated_tasks[task.id].update({
                "status": task.status.state.name.lower().replace("task_state_", ""),
                "timestamp": str(task.status.timestamp) or "now",
            })
            if task.status.message:
                content = []
                for part in task.status.message.parts:
                    if part.root.kind == "text":
                        content.append(part.root.text)
                if content:
                    self._delegated_tasks[task.id]["result"] = "\n".join(content)

    def get_agent_card(self, base_url: str):
        """Get an A2A AgentCard for this lead.

        Args:
            base_url: The base URL where this agent is hosted.

        Returns:
            An A2A AgentCard.
        """
        return persona_to_agent_card(self.config, base_url)

    async def _init_member_connections(self) -> None:
        """Initialize connections to all team members."""
        member_info = []

        for member_type in self.members:
            try:
                persona = self.persona_loader.load(member_type)
                if persona.url:
                    await self.connections.connect(persona.url)
                member_info.append({
                    "name": persona.name,
                    "description": persona.description,
                })
            except Exception:
                # Member not available yet
                pass

        self._members_info = "\n".join(json.dumps(info) for info in member_info)

    def create_agent(self) -> ADKAgent:
        """Create the ADK agent for this lead.

        Returns:
            The configured ADK Agent.
        """
        if self._adk_agent is not None:
            return self._adk_agent

        # Enable LiteLLM debug mode
        import litellm

        # Use model name as-is (should already include provider prefix like openai/)
        model = LiteLlm(model=self.config.model)

        # Configure for streaming responses
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.config.temperature,
        )

        # Build tools list: always include delegation tools
        tools: list[Any] = [
            self.list_team_members,
            self.delegate_to_member,
        ]

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
                tools.append(toolset)
                _logger.info(f"Loaded MCP toolset: {tool_name}")

        self._adk_agent = ADKAgent(
            name=self.config.name.lower().replace(" ", "_"),
            model=model,
            description=self.config.description,
            instruction=self._lead_instruction,
            before_model_callback=self._before_model,
            generate_content_config=generate_config,
            tools=tools,
        )
        return self._adk_agent

    def _lead_instruction(self, context: ReadonlyContext) -> str:
        """Build instruction for the lead agent."""
        current_member = self._check_state(context)
        
        # Use instructions from YAML
        base_instruction = self.config.instructions
        
        return f"""{base_instruction}

Team Coordination:
- Use `list_team_members` to see your team members and their capabilities
- Use `delegate_to_member` to assign work to a team member (asynchronously)

Available Team Members:
{self._members_info}

Current active member: {current_member['active_member']}
"""

    def _check_state(self, context: ReadonlyContext) -> dict[str, Any]:
        """Check the current conversation state."""
        state = context.state
        if (
            "context_id" in state
            and "session_active" in state
            and state["session_active"]
            and "member" in state
        ):
            return {"active_member": state["member"]}
        return {"active_member": "None"}

    def _before_model(
        self, callback_context: CallbackContext, llm_request: Any
    ) -> None:
        """Callback before model invocation."""
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            state["session_active"] = True

    def list_team_members(self) -> list[dict[str, str]]:
        """List available team members.

        Returns:
            List of member info with name and description.
        """
        result = []

        for member_type in self.members:
            try:
                persona = self.persona_loader.load(member_type)
                result.append({
                    "name": persona.name,
                    "description": persona.description,
                })
            except Exception:
                pass

        return result

    async def delegate_to_member(
        self,
        member_name: str,
        message: str,
        tool_context: ToolContext,
    ) -> list[Any]:
        """Delegate a task to a team member and return their response.

        Args:
            member_name: Name of the team member to delegate to.
            message: The message/task to send.
            tool_context: The ADK tool context.

        Returns:
            The response from the team member.
        """
        state = tool_context.state
        state["member"] = member_name

        # Find the connection
        agent = self.connections.get(member_name)
        if not agent:
            # Try to find by persona name
            for member_type in self.members:
                try:
                    persona = self.persona_loader.load(member_type)
                    if persona.name == member_name and persona.url:
                        agent = await self.connections.connect(persona.url)
                        break
                except Exception:
                    pass

        if not agent:
            raise ValueError(f"Team member '{member_name}' not found or not available")

        # Get or create context for this member
        member_contexts = state.get("member_contexts", {})
        member_context = member_contexts.get(member_name, {})
        context_id = member_context.get("context_id")
        task_id = member_context.get("task_id")

        # Use callback URL from config or default
        webhook_url = f"{self.webhook_base_url}/webhook/task"
        
        # Send message asynchronously
        _logger.info(f"Sending async message to {agent.name} with webhook {webhook_url}")
        
        task_id = await agent.send_message_async(
            text=message,
            webhook_url=webhook_url,
            context_id=context_id,
            task_id=task_id,
            callback=self._on_task_update,
        )
        
        # Track the delegated task
        self._delegated_tasks[task_id] = {
            "agent": agent.name,
            "status": "submitted",
            "message": message[:50] + "...",
            "timestamp": "now",  # Should use real timestamp
        }

        # Update context
        member_context["task_id"] = task_id
        member_contexts[member_name] = member_context
        state["member_contexts"] = member_contexts
        
        return [{
            "status": "submitted",
            "task_id": task_id,
            "agent": agent.name,
            "message": "Task dispatched successfully. Wait for push notification.",
            "timestamp": "now"
        }]

    def get_delegated_tasks(self) -> dict[str, dict[str, Any]]:
        """Get all delegated tasks and their current status."""
        return self._delegated_tasks.copy()

    def _extract_parts(self, parts: list[Part]) -> list[Any]:
        """Extract content from message parts."""
        result = []
        for part in parts:
            if part.root.kind == "text":
                result.append(part.root.text)
            elif part.root.kind == "data":
                result.append(part.root.data)
        return result

    async def get_member_status(self) -> dict[str, Any]:
        """Get status of all team members with retry and exponential backoff.

        Returns:
            Dictionary with member health status.
        """
        async def check_member(member_type: str) -> tuple[str, dict[str, Any]]:
            """Check a single member's health with retry."""
            try:
                persona = self.persona_loader.load(member_type)
                url = persona.url
                if not url:
                    return member_type, {
                        "healthy": False,
                        "error": "no URL configured",
                        "url": "",
                    }

                # Probe with retry and exponential backoff
                success, status_code, error = await http_get_with_retry(
                    self.httpx_client,
                    f"{url}/health",
                    timeout=5.0,
                    max_retries=3,
                    base_delay=0.5,
                )

                if success:
                    return member_type, {
                        "healthy": status_code == 200,
                        "error": "" if status_code == 200 else f"HTTP {status_code}",
                        "url": url,
                    }
                return member_type, {
                    "healthy": False,
                    "error": error,
                    "url": url,
                }
            except FileNotFoundError:
                return member_type, {
                    "healthy": False,
                    "error": "persona not found",
                    "url": "",
                }

        # Check all members in parallel
        member_checks = [check_member(m) for m in self.members]
        results_list = await asyncio.gather(*member_checks, return_exceptions=True)

        results: dict[str, Any] = {}
        for result in results_list:
            if isinstance(result, tuple):
                member_type, status = result
                results[member_type] = status
            elif isinstance(result, Exception):
                # Shouldn't happen, but handle gracefully
                pass

        return results
