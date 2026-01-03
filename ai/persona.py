"""Persona is a YAML based configuration for an agent."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from pathlib import Path
from pydantic_yaml import parse_yaml_file_as


class Persona(BaseModel):
    """Persona is a YAML based configuration for an agent."""
    name: str = ""
    description: str = ""
    instructions: str = ""
    model: str = ""
    temperature: float = 0.0
    # List of MCP tool server names / toolset names enabled for this persona.
    # (Matches `config/personas/*.yml`.)
    tools: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_yaml(yaml_path: Path) -> Persona:
        """Load a persona from a YAML file."""
        return parse_yaml_file_as(Persona, yaml_path)