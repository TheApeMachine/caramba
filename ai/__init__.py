"""AI Agent System for Caramba.

This package provides a multi-agent system using Google's Agent Development Kit (ADK)
with A2A (Agent-to-Agent) protocol for inter-agent communication and MCP (Model Context
Protocol) for tool integration.

Architecture:
- Root agent: The only agent that communicates with users, delegates to team leads
- Teams: Groups of specialized agents under a lead (defined in config/teams/)
- Personas: Agent configurations loaded from YAML (defined in config/personas/)

Key Components:
- Agent: ADK agent wrapper with persona configuration
- RootAgent: Orchestrator that delegates to team leads
- LeadAgent: Team lead that delegates to members
- AgentServer: A2A-compatible HTTP server for agents
- ConnectionManager: Manages connections to remote agents
- ADKAgentExecutor: Bridges ADK with A2A protocol
"""
from ai.agent import Agent, AgentFactory
from ai.connection import ConnectionManager, RemoteAgent
from ai.executor import ADKAgentExecutor, StreamingExecutor
from ai.lead import LeadAgent
from ai.persona import PersonaLoader, persona_to_agent_card
from ai.root import RootAgent
from ai.server import (
    AgentServer,
    run_root_server,
    run_lead_server,
    run_agent_server,
)
from ai.session_store import (
    DatabaseSessionService,
    get_shared_session_service,
)
from .retry import (
    calculate_backoff,
    http_get_json_with_retry,
    http_get_with_retry,
    retry_async,
)
from .team import TeamLoader, TeamRegistry
from .types import (
    AgentHealth,
    AgentState,
    PersonaConfig,
    TeamConfig,
    TeamHealth,
)

__all__ = [
    # Core agent classes
    "Agent",
    "AgentFactory",
    "RootAgent",
    "LeadAgent",
    # Execution
    "ADKAgentExecutor",
    "StreamingExecutor",
    # Server
    "AgentServer",
    "run_root_server",
    "run_lead_server",
    "run_agent_server",
    # Connections
    "ConnectionManager",
    "RemoteAgent",
    # Configuration
    "PersonaLoader",
    "TeamLoader",
    "TeamRegistry",
    "persona_to_agent_card",
    # Types
    "AgentHealth",
    "AgentState",
    "PersonaConfig",
    "TeamConfig",
    "TeamHealth",
    # Retry utilities
    "calculate_backoff",
    "http_get_json_with_retry",
    "http_get_with_retry",
    "retry_async",
    # Session persistence
    "DatabaseSessionService",
    "get_shared_session_service",
]
