from __future__ import annotations

from agent.persona import Persona
from config.persona import DeveloperConfig, PersonaType


def test_persona_wraps_config_fields() -> None:
    cfg = DeveloperConfig(
        type=PersonaType.DEVELOPER,
        name="Dev",
        description="d",
        instructions="i",
        model="m",
        temperature=0.3,
        tool_choice="auto",
        mcp_servers=["graphiti"],
    )
    p = Persona(cfg)
    assert p.name() == "Dev"
    assert p.model() == "m"
    assert p.instructions() == "i"
    assert p.temperature() == 0.3
    assert p.tool_choice() == "auto"
    assert p.mcp_servers() == ["graphiti"]

