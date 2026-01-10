"""Core types for the AI agent system.

Defines the foundational data structures used throughout the agent system,
including persona configuration, team topology, and agent state management.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    """Lifecycle states for an agent following A2A protocol."""

    IDLE = "idle"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PersonaConfig:
    """Configuration for an agent persona loaded from YAML.

    Defines the identity, behavior, and capabilities of an agent.
    """

    name: str
    type: str
    description: str
    instructions: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    tool_choice: str = "auto"
    tools: list[str] = field(default_factory=list)
    sub_agents: list[str] = field(default_factory=list)
    url: str | None = None
    version: str = "1.0.0"
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> PersonaConfig:
        """Create a PersonaConfig from a dictionary."""
        return cls(
            name=data.get("name", ""),
            type=data.get("type", ""),
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
        )
        

@dataclass
class TeamConfig:
    """Configuration for a team of agents.

    Teams group specialized agents together under a lead.
    """

    name: str
    lead: str
    members: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> TeamConfig:
        """Create a TeamConfig from a dictionary."""
        return cls(
            name=name,
            lead=data.get("lead", ""),
            members=data.get("members", []),
        )


@dataclass
class AgentHealth:
    """Health status of an agent."""

    name: str
    healthy: bool
    error: str = ""
    url: str = ""
    activity: AgentState = AgentState.IDLE


@dataclass
class TeamHealth:
    """Health status of a team including all members."""

    name: str
    lead: str
    agents: dict[str, AgentHealth] = field(default_factory=dict)
