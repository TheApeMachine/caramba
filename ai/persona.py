"""Persona is a YAML based configuration for an agent."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import yaml


@dataclass
class Persona:
    """Persona is a YAML based configuration for an agent."""
    name: str = ""
    description: str = ""
    instructions: str = ""
    model: str = ""
    temperature: float = 0.0
    # List of MCP tool server names / toolset names enabled for this persona.
    # (Matches `config/personas/*.yml`.)
    tools: list[str] = field(default_factory=list)

    @staticmethod
    def from_yaml(yaml_path: Path) -> Persona:
        """Load a persona from a YAML file."""
        with open(yaml_path, 'r') as f:
            payload = yaml.safe_load(f) or {}
            if not isinstance(payload, dict):
                raise TypeError(f"Persona YAML must be a mapping, got {type(payload)!r}")

            # Be forgiving: persona YAMLs can include extra keys (e.g. `type:` from config/persona.py).
            allowed = {fld.name for fld in fields(Persona)}
            filtered = {k: v for k, v in payload.items() if k in allowed}
            return Persona(**filtered)