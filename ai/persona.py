"""Persona is a YAML based configuration for an agent."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from pathlib import Path
from pydantic_yaml import parse_yaml_file_as


class ToolRef(BaseModel):
    """Tool reference used in persona YAML.

    Supports the common format:
      tools:
        - name: "filesystem"
    """

    name: str = Field(default="", description="Tool/server name")


class Persona(BaseModel):
    """Persona is a YAML based configuration for an agent."""
    type: str = Field(default="", description="Stable identifier for the persona (YAML `type:`)")
    name: str = Field(default="", description="Name of the agent persona")
    description: str = Field(default="", description="Description of the agent persona")
    instructions: str = Field(default="", description="System instructions for the agent")
    model: str = Field(default="", description="Model identifier to use for this persona")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
    # List of MCP tool server names / toolset names enabled for this persona.
    # (Matches `config/personas/*.yml`.)
    tools: list[str | ToolRef] = Field(
        default_factory=list,
        description="List of MCP tool server names enabled for this persona",
    )
    # Optional list of sub-agent names (for A2A delegation).
    # Empty by default; only root persona will have sub-agents initially.
    sub_agents: list[str] = Field(default_factory=list, description="List of sub-agent persona names (A2A)")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for structured output")

    @staticmethod
    def from_yaml(yaml_path: Path) -> Persona:
        """Load a persona from a YAML file."""
        return parse_yaml_file_as(Persona, yaml_path)