"""Persona loader for agents.

Loads persona configurations from YAML files in config/personas/.
Uses the config.persona module for the actual configuration classes.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from config.persona import PersonaConfig, PersonaType, SharedPersonaConfig


class Persona:
    """Persona of an agent"""
    def __init__(self, persona: PersonaConfig):
        self.config = persona

    def instructions(self) -> str:
        return self.config.instructions

    def model(self) -> str:
        return self.config.model

    def temperature(self) -> float:
        return self.config.temperature

    def tool_choice(self) -> str:
        return self.config.tool_choice

    def mcp_servers(self) -> list[str]:
        return self.config.mcp_servers

    def name(self) -> str:
        return self.config.name
