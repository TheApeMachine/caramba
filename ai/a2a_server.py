"""A2A server for exposing personas as remote agents.

This module provides a generic A2A server that loads a persona YAML and
exposes it via ADK's A2A protocol so other agents can call it remotely.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.console import logger


def main() -> None:
    """Main entrypoint for persona A2A server.

    Expects environment variables:
    - PERSONA_FILE: Path to persona YAML (relative to repo root or absolute)
    - PORT: Port to listen on (default: 8001)
    """
    persona_file = os.getenv("PERSONA_FILE")

    if not persona_file:
        logger.error("PERSONA_FILE environment variable is required")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    persona_path = Path(persona_file)

    if not persona_path.is_absolute():
        persona_path = repo_root / persona_path

    persona = Persona.from_yaml(persona_path)
    agent = Agent(persona=persona)

    port = int(os.getenv("PORT", "8001"))

    logger.info(f"Starting A2A server for {persona_path.name} on port {port}")
    logger.info(f"Agent card available at: http://0.0.0.0:{port}/.well-known/agent-card.json")
    logger.info(f"Health check at: http://0.0.0.0:{port}/health")
    logger.info(f"Agent status at: http://0.0.0.0:{port}/agents/status")

    uvicorn.run(agent.server(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
