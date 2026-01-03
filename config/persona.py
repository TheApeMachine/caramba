"""Persona configuration for agents"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, Self, TypeAlias
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from caramba.config import Config


class PersonaType(str, enum.Enum):
    """Type of persona"""
    RESEARCH_LEAD = "research_lead"
    WRITER = "writer"
    REVIEWER = "reviewer"
    MACHINE_LEARNING_EXPERT = "machine_learning_expert"
    MATHEMATICIAN = "mathematician"
    DEVELOPER = "developer"
    CATALYST = "catalyst"
    ARCHITECT = "architect"

    @classmethod
    def from_str(cls, s: str) -> "PersonaType":
        return cls(s)

    @classmethod
    def module_name(cls) -> str:
        # The agent system is being migrated to `caramba.ai`.
        return "caramba.ai.persona"


class SharedPersonaConfig(Config):
    """Shared persona configuration"""
    name: str
    description: str
    instructions: str
    model: str
    temperature: float
    # `tool_choice` existed in the older OpenAI Agents SDK wiring; keep it optional.
    tool_choice: str = "auto"

    # Tool server names.
    #
    # Historical configs used `mcp_servers: [...]`.
    # Newer configs (e.g. chatgpt/claude/gemini) used `tools: [...]`.
    #
    # We support BOTH and normalize them to stay compatible while the repo migrates.
    mcp_servers: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_tool_fields(self) -> Self:
        # If only one of the two lists is provided, mirror it into the other.
        if not self.mcp_servers and self.tools:
            self.mcp_servers = list(self.tools)
        elif not self.tools and self.mcp_servers:
            self.tools = list(self.mcp_servers)
        elif self.mcp_servers and self.tools and self.mcp_servers != self.tools:
            raise ValueError(
                f"Both mcp_servers and tools are provided but differ: "
                f"mcp_servers={self.mcp_servers}, tools={self.tools}. "
                f"Please reconcile these values."
            )
        return self


class ResearchLeadConfig(SharedPersonaConfig):
    """Research lead configuration"""
    type: Literal[PersonaType.RESEARCH_LEAD] = PersonaType.RESEARCH_LEAD
    mcp_servers: list[str] = Field(default_factory=lambda: ["graphiti"])


class WriterConfig(SharedPersonaConfig):
    """Writer configuration"""
    type: Literal[PersonaType.WRITER] = PersonaType.WRITER


class ReviewerConfig(SharedPersonaConfig):
    """Reviewer configuration"""
    type: Literal[PersonaType.REVIEWER] = PersonaType.REVIEWER


class MachineLearningExpertConfig(SharedPersonaConfig):
    """Machine learning expert configuration"""
    type: Literal[PersonaType.MACHINE_LEARNING_EXPERT] = PersonaType.MACHINE_LEARNING_EXPERT


class MathematicianConfig(SharedPersonaConfig):
    """Mathematician configuration"""
    type: Literal[PersonaType.MATHEMATICIAN] = PersonaType.MATHEMATICIAN


class DeveloperConfig(SharedPersonaConfig):
    """Developer configuration"""
    type: Literal[PersonaType.DEVELOPER] = PersonaType.DEVELOPER


class CatalystConfig(SharedPersonaConfig):
    """Catalyst configuration"""
    type: Literal[PersonaType.CATALYST] = PersonaType.CATALYST


class ArchitectConfig(SharedPersonaConfig):
    """Architect configuration (platform design + technical planning)."""
    type: Literal[PersonaType.ARCHITECT] = PersonaType.ARCHITECT


# Union type for any persona config, with automatic deserialization
PersonaConfig: TypeAlias = Annotated[
    ResearchLeadConfig
    | WriterConfig
    | ReviewerConfig
    | MachineLearningExpertConfig
    | MathematicianConfig
    | DeveloperConfig
    | CatalystConfig
    | ArchitectConfig,
    Field(discriminator="type"),
]


def load_persona(name: str, personas_dir: Path = Path("config/personas")) -> PersonaConfig:
    """Load a persona configuration from a YAML file.

    Args:
        name: Name of the persona (e.g., 'research_lead').
        personas_dir: Directory containing persona YAML files.

    Returns:
        The validated persona configuration.
    """
    path = personas_dir / f"{name}.yml"
    if not path.exists():
        # Fallback to research_lead if name doesn't match
        path = personas_dir / "research_lead.yml"

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Wrap in TypeAdapter to handle Union type validation
    from pydantic import TypeAdapter
    return TypeAdapter(PersonaConfig).validate_python(data)
