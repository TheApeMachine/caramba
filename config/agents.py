"""Agent process configuration.

This extends the manifest-driven approach to the built-in agent system:
- define a *team* (keys -> persona yaml names)
- define one or more *processes* (discussion, paper workflows, etc.)
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, RootModel


class AgentTeamConfig(RootModel[dict[str, str]]):
    """Mapping of team role keys to persona names.

    Example:
        team:
          research_team_leader: research_lead
          developer: developer
    """


class DiscussionProcessConfig(BaseModel):
    """Configuration for the `discussion` agent process."""

    type: Literal["discussion"] = "discussion"
    name: str
    leader: str
    topic: str
    prompts_dir: str = "config/prompts"
    max_rounds: int = Field(default=12, ge=1)


AgentProcessConfig: TypeAlias = Annotated[
    DiscussionProcessConfig,
    Field(discriminator="type"),
]


class AgentsConfig(BaseModel):
    """Top-level agent section inside a manifest."""

    team: AgentTeamConfig
    processes: list[AgentProcessConfig] = Field(default_factory=list)

