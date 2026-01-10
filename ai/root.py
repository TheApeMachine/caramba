"""Root agent that delegates to team leads.

The root agent is the only agent that communicates directly with users.
It has no tools of its own but can delegate to team leads who then
coordinate with their team members.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import httpx

# Set up file logging for debugging
# Try to log to file, fall back to console only if file not accessible
_logger = logging.getLogger("caramba.root")
_logger.setLevel(logging.DEBUG)
try:
    _log_handler = logging.FileHandler("/app/artifacts/root_agent.log")
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(_log_handler)
except Exception:
    pass  # File not accessible (e.g., local dev without Docker)
# Also add console handler for docker logs
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
_logger.addHandler(_console)

# Configure the connection logger to also write to the same file
_conn_logger = logging.getLogger("caramba.connection")
_conn_logger.setLevel(logging.DEBUG)
for handler in _logger.handlers:
    _conn_logger.addHandler(handler)
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
from .retry import http_get_json_with_retry, http_get_with_retry
from .team import TeamLoader, TeamRegistry
from .types import AgentState


# Callback to notify the user of task updates (set by server)
_user_notification_callback: TaskCallback | None = None


def set_user_notification_callback(callback: TaskCallback | None) -> None:
    """Set the callback for notifying users of async task updates."""
    global _user_notification_callback
    _user_notification_callback = callback


class RootAgent:
    """The root orchestrator agent.

    Communicates with users and delegates tasks to team leads.
    Does not have any tools of its own.
    """

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        team_config: str = "default",
        webhook_base_url: str | None = None,
    ) -> None:
        """Initialize the root agent.

        Args:
            httpx_client: The httpx client for A2A connections.
            team_config: The team configuration file name.
            webhook_base_url: Base URL for this agent's webhook (for push notifications).
        """
        self.httpx_client = httpx_client
        self.team_config = team_config
        self.webhook_base_url = webhook_base_url or "http://root-agent:8001"
        self.persona_loader = PersonaLoader()
        self.team_loader = TeamLoader()
        self.registry = TeamRegistry(self.team_loader)
        self.connections = ConnectionManager(httpx_client)

        # Load root persona
        self.config = self.persona_loader.load("root")
        self._leads_info: str = ""
        self._adk_agent: ADKAgent | None = None

        # Track delegated tasks for async updates
        self._delegated_tasks: dict[str, dict[str, Any]] = {}

        # Initialize connections to leads in background
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._init_lead_connections())

    async def _on_task_update(self, task: Task) -> None:
        """Handle updates for delegated tasks."""
        _logger.info(f"RootAgent received update for task {task.id}: {task.status.state}")
        
        # Update local tracking
        if task.id in self._delegated_tasks:
            self._delegated_tasks[task.id].update({
                "status": task.status.state.name.lower().replace("task_state_", ""),
                "timestamp": str(task.status.timestamp) or "now",
            })
            if task.status.message:
                # Extract text from message parts
                content = []
                for part in task.status.message.parts:
                    if part.root.kind == "text":
                        content.append(part.root.text)
                if content:
                    self._delegated_tasks[task.id]["result"] = "\n".join(content)

        # Notify global listener if set
        if _user_notification_callback:
            await _user_notification_callback(task)

    def get_agent_card(self, base_url: str):
        """Get an A2A AgentCard for the root agent.

        Args:
            base_url: The base URL where this agent is hosted.

        Returns:
            An A2A AgentCard.
        """
        return persona_to_agent_card(self.config, base_url)

    async def _init_lead_connections(self) -> None:
        """Initialize connections to all team leads."""
        leads = self.team_loader.get_leads(self.team_config)
        _logger.info(f"Initializing connections to leads: {leads}")
        lead_info = []

        async def try_connect(lead_type: str) -> dict[str, str] | None:
            """Try to connect to a lead with retry."""
            try:
                persona = self.persona_loader.load(lead_type)
                _logger.debug(f"Loaded persona {lead_type}: url={persona.url}")
                if persona.url:
                    # Retry connection up to 3 times with backoff
                    for attempt in range(3):
                        try:
                            _logger.info(f"Connecting to {lead_type} at {persona.url} (attempt {attempt+1})")
                            await self.connections.connect(persona.url)
                            self.registry.register_agent(lead_type, persona.url)
                            _logger.info(f"Successfully connected to {lead_type}")
                            return {
                                "name": persona.name,
                                "description": persona.description,
                            }
                        except Exception as e:
                            _logger.warning(f"Connection attempt {attempt+1} to {lead_type} failed: {e}")
                            if attempt < 2:
                                await asyncio.sleep(0.5 * (2 ** attempt))
                return {"name": persona.name, "description": persona.description}
            except Exception as e:
                _logger.error(f"Failed to load persona {lead_type}: {e}")
                return None

        # Connect to all leads in parallel
        results = await asyncio.gather(
            *[try_connect(lead_type) for lead_type in leads],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, dict):
                lead_info.append(result)

        _logger.info(f"Lead connections initialized. Connected: {list(self.connections._connections.keys())}")

        self._leads_info = "\n".join(json.dumps(info) for info in lead_info)

    def create_agent(self) -> ADKAgent:
        """Create the ADK agent for the root.

        Returns:
            The configured ADK Agent.
        """
        if self._adk_agent is not None:
            return self._adk_agent

        # Enable LiteLLM debug mode
        import litellm

        # Use model name as-is (should already include provider prefix like openai/)
        model = LiteLlm(model=self.config.model)
        _logger.info(f"Creating root agent with model: {self.config.model}")

        # Configure for streaming responses
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.config.temperature,
        )

        self._adk_agent = ADKAgent(
            name="root_agent",
            model=model,
            description=self.config.description,
            instruction=self._root_instruction,
            before_model_callback=self._before_model,
            generate_content_config=generate_config,
            tools=[
                self.list_team_leads,
                self.delegate_to_lead,
            ],
        )
        return self._adk_agent

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """Build instruction for the root agent."""
        current_lead = self._check_state(context)
        
        # Use instructions from YAML
        base_instruction = self.config.instructions
        
        # Add dynamic state information
        return f"""{base_instruction}

Discovery:
- Use `list_team_leads` to see available team leads and their capabilities

Execution:
- Use `delegate_to_lead` to send a task to a team lead
- Include context about what you need from them

Available Team Leads:
{self._leads_info}

Current active lead: {current_lead['active_lead']}
"""

    def _check_state(self, context: ReadonlyContext) -> dict[str, Any]:
        """Check the current conversation state."""
        state = context.state
        if (
            "context_id" in state
            and "session_active" in state
            and state["session_active"]
            and "lead" in state
        ):
            return {"active_lead": state["lead"]}
        return {"active_lead": "None"}

    def _before_model(
        self, callback_context: CallbackContext, llm_request: Any
    ) -> None:
        """Callback before model invocation."""
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            state["session_active"] = True

    def list_team_leads(self) -> list[dict[str, str]]:
        """List available team leads for delegation.

        Returns:
            List of lead info with name and description.
        """
        leads = self.team_loader.get_leads(self.team_config)
        result = []

        for lead_type in leads:
            try:
                persona = self.persona_loader.load(lead_type)
                result.append({
                    "name": persona.name,
                    "description": persona.description,
                    "teams": [
                        t.name for t in self.team_loader.get_teams_for_agent(lead_type)
                    ],
                })
            except Exception:
                pass

        return result

    async def delegate_to_lead(
        self,
        lead_name: str,
        message: str,
        tool_context: ToolContext,
    ) -> list[Any]:
        """Delegate a task to a team lead and return their response.

        Args:
            lead_name: Name of the team lead to delegate to.
            message: The message/task to send.
            tool_context: The ADK tool context.

        Returns:
            The response from the team lead.
        """
        state = tool_context.state
        state["lead"] = lead_name

        _logger.info(f"delegate_to_lead called: lead_name='{lead_name}', message='{message[:100]}...'")
        _logger.debug(f"Available connections: {list(self.connections._connections.keys())}")

        # Try multiple ways to find the agent
        agent = self._find_agent(lead_name)

        if not agent:
            # Try to connect by finding the persona
            agent = await self._connect_to_lead(lead_name)

        if not agent:
            # List available leads for debugging
            available = list(self.connections._connections.keys())
            _logger.error(f"Failed to find agent '{lead_name}'. Available: {available}")
            raise ValueError(
                f"Team lead '{lead_name}' not found. Available: {available}"
            )

        state["current_lead"] = agent.name

        # Get or create context for this lead
        lead_contexts = state.get("lead_contexts", {})
        lead_context = lead_contexts.get(agent.name, {})
        context_id = lead_context.get("context_id")
        task_id = lead_context.get("task_id")

        # Send message and wait for response
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
        lead_context["task_id"] = task_id
        lead_contexts[agent.name] = lead_context
        state["lead_contexts"] = lead_contexts
        
        return [{
            "status": "submitted",
            "task_id": task_id,
            "agent": agent.name,
            "message": "Task dispatched successfully. Wait for push notification.",
            "timestamp": "now"
        }]

    def _find_agent(self, lead_name: str):
        """Find an agent by name with various matching strategies."""
        agent = self.connections.get(lead_name)

        if not agent:
            # Try case-insensitive and partial match
            for name in list(self.connections._connections.keys()):
                if name.lower() == lead_name.lower():
                    agent = self.connections.get(name)
                    _logger.info(f"Found by case-insensitive match: {name}")
                    break
                # Try matching underscored version
                if name.lower().replace("_", " ") == lead_name.lower().replace("_", " "):
                    agent = self.connections.get(name)
                    _logger.info(f"Found by underscore match: {name}")
                    break

        return agent

    async def _connect_to_lead(self, lead_name: str):
        """Try to connect to a lead by finding the persona."""
        _logger.debug("No direct match, trying persona lookup")
        for lead_type in self.team_loader.get_leads(self.team_config):
            try:
                persona = self.persona_loader.load(lead_type)
                _logger.debug(f"Checking persona: type={lead_type}, name={persona.name}")
                # Match by name (case-insensitive) or by persona type
                if (
                    persona.name.lower() == lead_name.lower()
                    or persona.name.lower().replace("_", " ") == lead_name.lower().replace("_", " ")
                    or lead_type.lower() == lead_name.lower()
                ):
                    if persona.url:
                        _logger.info(f"Connecting to {persona.url}")
                        agent = await self.connections.connect(persona.url)
                        _logger.info("Connected successfully")
                        return agent
            except Exception as e:
                _logger.error(f"Error connecting to {lead_type}: {e}", exc_info=True)
        return None

    def get_delegated_tasks(self) -> dict[str, dict[str, Any]]:
        """Get all delegated tasks and their current status."""
        return self._delegated_tasks.copy()

    def get_task_updates(self, task_id: str) -> dict[str, Any] | None:
        """Get updates for a specific delegated task."""
        return self._delegated_tasks.get(task_id)

    def _extract_parts(self, parts: list[Part]) -> list[Any]:
        """Extract content from message parts."""
        result = []
        for part in parts:
            if part.root.kind == "text":
                result.append(part.root.text)
            elif part.root.kind == "data":
                result.append(part.root.data)
        return result

    async def get_hierarchy_status(self) -> dict[str, Any]:
        """Get the full agent hierarchy with health status.

        Queries each team lead for their member status.

        Returns:
            Dictionary suitable for /agents/status endpoint.
        """
        teams = self.registry.build_hierarchy(self.team_config)

        # Query each team lead for their member status with retry
        async def query_lead_members(lead_url: str) -> dict[str, Any]:
            """Query a lead agent for its members' status with retry."""
            if not lead_url:
                return {}

            success, data, _ = await http_get_json_with_retry(
                self.httpx_client,
                f"{lead_url}/members/status",
                timeout=10.0,
                max_retries=3,
                base_delay=0.5,
            )
            if success and data:
                return data.get("members", {})
            return {}

        # Check lead health with retry and query their members
        async def check_lead_health(lead_type: str) -> tuple[str, bool, str, dict[str, Any]]:
            """Check lead health and get member status with retry."""
            try:
                persona = self.persona_loader.load(lead_type)
                url = persona.url
                if not url:
                    return lead_type, False, "no URL configured", {}

                # Check lead's health with retry
                success, status_code, error = await http_get_with_retry(
                    self.httpx_client,
                    f"{url}/health",
                    timeout=5.0,
                    max_retries=3,
                    base_delay=0.5,
                )

                if not success:
                    return lead_type, False, error, {}

                lead_healthy = status_code == 200
                lead_error = "" if lead_healthy else f"HTTP {status_code}"

                # Query lead for member status (also with retry)
                members = await query_lead_members(url)
                return lead_type, lead_healthy, lead_error, members

            except FileNotFoundError:
                return lead_type, False, "persona not found", {}

        # Get all leads and check them in parallel
        leads = self.team_loader.get_leads(self.team_config)
        lead_results = await asyncio.gather(
            *[check_lead_health(lead) for lead in leads],
            return_exceptions=True,
        )

        # Build health maps
        lead_health: dict[str, tuple[bool, str]] = {}
        member_health: dict[str, tuple[bool, str]] = {}

        for result in lead_results:
            if isinstance(result, tuple):
                lead_type, healthy, error, members = result
                lead_health[lead_type] = (healthy, error)
                for member_name, member_info in members.items():
                    if isinstance(member_info, dict):
                        member_health[member_name] = (
                            member_info.get("healthy", False),
                            member_info.get("error", ""),
                        )

        # Update team info with health from lead and member checks
        for team_name, team_info in teams.items():
            for agent_name, agent_info in team_info["agents"].items():
                # Check if this is a lead
                if agent_name in lead_health:
                    healthy, error = lead_health[agent_name]
                    agent_info["healthy"] = healthy
                    agent_info["error"] = error
                # Check if this is a member (from any lead's report)
                elif agent_name in member_health:
                    healthy, error = member_health[agent_name]
                    agent_info["healthy"] = healthy
                    agent_info["error"] = error

        return {
            "root": {
                "name": self.config.name,
                "healthy": True,
            },
            "teams": teams,
        }
