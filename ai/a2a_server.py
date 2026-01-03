"""A2A server for exposing personas as remote agents.

This module provides a generic A2A server that loads a persona YAML and
exposes it via ADK's A2A protocol so other agents can call it remotely.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.console import logger


def create_a2a_server_for_persona(persona_path: Path, *, port: int = 8001) -> None:
    """Create and run an A2A server for a single persona.

    Args:
        persona_path: Path to persona YAML file.
        port: Port to listen on (default: 8001).
    """
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")

    persona = Persona.from_yaml(persona_path)
    logger.info(f"Loading persona: {persona.name} from {persona_path}")

    # Create an Agent instance (this handles model selection, MCP tools, etc.)
    agent = Agent(persona=persona)

    # Wrap agent with A2A and create FastAPI app.
    # Note: to_a2a returns a FastAPI app that exposes the agent via A2A protocol.
    a2a_app = to_a2a(agent.adk_agent, port=port)

    logger.info(f"Starting A2A server for {persona_path.name} on port {port}")
    logger.info(f"Agent card available at: http://0.0.0.0:{port}/.well-known/agent-card.json")

    # Run the server (blocks until shutdown).
    # Use asyncio.run for compatibility with ADK's async server.
    import uvicorn
    uvicorn.run(a2a_app, host="0.0.0.0", port=port, log_level="info")


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

    # Resolve persona path (support both relative and absolute paths).
    repo_root = Path(__file__).resolve().parent.parent
    persona_path = Path(persona_file)
    if not persona_path.is_absolute():
        persona_path = repo_root / persona_path

    port = int(os.getenv("PORT", "8001"))

    create_a2a_server_for_persona(persona_path, port=port)


if __name__ == "__main__":
    main()
