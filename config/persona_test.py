from __future__ import annotations

from pathlib import Path

from caramba.config.persona import PersonaType, load_persona


def test_persona_type_from_str_roundtrip() -> None:
    assert PersonaType.from_str("developer") == PersonaType.DEVELOPER
    assert PersonaType.DEVELOPER.value == "developer"


def test_load_persona_loads_and_validates(tmp_path: Path) -> None:
    personas = tmp_path / "personas"
    personas.mkdir()

    (personas / "developer.yml").write_text(
        "\n".join(
            [
                "type: developer",
                "name: Dev",
                "description: Developer persona",
                "instructions: Write good code",
                "model: gpt-4.1-mini",
                "temperature: 0.2",
                "tool_choice: auto",
                "mcp_servers: [cursor-ide-browser]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_persona("developer", personas_dir=personas)
    assert cfg.type == PersonaType.DEVELOPER
    assert cfg.name == "Dev"
    assert cfg.model == "gpt-4.1-mini"
    assert cfg.mcp_servers == ["cursor-ide-browser"]


def test_load_persona_falls_back_to_research_lead(tmp_path: Path) -> None:
    personas = tmp_path / "personas"
    personas.mkdir()

    (personas / "research_lead.yml").write_text(
        "\n".join(
            [
                "type: research_lead",
                "name: Lead",
                "description: Lead persona",
                "instructions: Lead the research",
                "model: gpt-4.1",
                "temperature: 0.1",
                "tool_choice: auto",
                # Uses default_factory in config, but include explicitly to keep test simple.
                "mcp_servers: [graphiti]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_persona("does_not_exist", personas_dir=personas)
    assert cfg.type == PersonaType.RESEARCH_LEAD
    assert cfg.name == "Lead"

