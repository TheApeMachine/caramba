"""Persona management for loading agent configurations.

Loads persona definitions from YAML files and creates AgentCard instances
for A2A protocol compatibility.
"""
from __future__ import annotations

from pathlib import Path
import yaml
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from ai.types import PersonaConfig


class PersonaLoader:
    """Loads and manages agent personas from YAML configuration files.

    Provides a central registry of all available personas and their
    configurations, enabling dynamic agent creation based on role.
    """

    def __init__(self, personas_dir: str | Path | None = None) -> None:
        """Initialize the persona loader.

        Args:
            personas_dir: Path to directory containing persona YAML files.
                         Defaults to config/personas, checking multiple locations.
        """
        if personas_dir is None:
            # Check multiple possible locations (Docker mount, local dev, package)
            candidates = [
                Path("/app/config/personas"),  # Docker volume mount
                Path(__file__).parent.parent / "config" / "personas",  # Package-relative
                Path.cwd() / "config" / "personas",  # Working directory
            ]
            for candidate in candidates:
                if candidate.exists():
                    personas_dir = candidate
                    break
            else:
                # Default to package-relative if none exist
                personas_dir = Path(__file__).parent.parent / "config" / "personas"
        self.personas_dir = Path(personas_dir)
        self._cache: dict[str, PersonaConfig] = {}

    def load(self, persona_type: str) -> PersonaConfig:
        """Load a persona configuration by type.

        Args:
            persona_type: The persona type (filename without extension).

        Returns:
            The loaded PersonaConfig.

        Raises:
            FileNotFoundError: If the persona file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        if persona_type in self._cache:
            return self._cache[persona_type]

        path = self.personas_dir / f"{persona_type}.yml"
        if not path.exists():
            raise FileNotFoundError(f"Persona not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid persona file: {path}")

        config = PersonaConfig.from_yaml(data)
        self._cache[persona_type] = config
        return config

    def load_all(self) -> dict[str, PersonaConfig]:
        """Load all available personas.

        Returns:
            Dictionary mapping persona type to PersonaConfig.
        """
        if not self.personas_dir.exists():
            return {}

        for path in self.personas_dir.glob("*.yml"):
            persona_type = path.stem
            if persona_type not in self._cache:
                self.load(persona_type)

        return self._cache.copy()

    def get_names(self) -> list[str]:
        """Get all available persona names.

        Returns:
            List of persona type names.
        """
        if not self.personas_dir.exists():
            return []
        return [p.stem for p in self.personas_dir.glob("*.yml")]


def persona_to_agent_card(config: PersonaConfig, base_url: str) -> AgentCard:
    """Convert a PersonaConfig to an A2A AgentCard.

    Args:
        config: The persona configuration.
        base_url: The base URL where this agent is hosted.

    Returns:
        An A2A-compatible AgentCard.
    """
    # AgentCapabilities uses snake_case in the a2a library
    # Enable streaming and push notifications for async operations
    capabilities = AgentCapabilities(
        streaming=config.capabilities.get("streaming", True),
        push_notifications=config.capabilities.get("push_notifications", True),
    )

    skills = []
    if config.capabilities.get("skills"):
        for skill_data in config.capabilities["skills"]:
            skills.append(
                AgentSkill(
                    id=skill_data.get("id", "chat"),
                    name=skill_data.get("name", "chat"),
                    description=skill_data.get("description", ""),
                    tags=skill_data.get("tags", []),
                    examples=skill_data.get("examples", []),
                )
            )
    else:
        skills.append(
            AgentSkill(
                id="chat",
                name="chat",
                description=config.description,
                tags=["chat"],
                examples=[],
            )
        )

    return AgentCard(
        name=config.name,
        description=config.description,
        url=config.url or base_url,
        version=config.version,
        capabilities=capabilities,
        skills=skills,
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
    )
