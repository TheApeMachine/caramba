"""A2A server for exposing personas as remote agents.

This module provides a generic A2A server that loads a persona YAML and
exposes it via ADK's A2A protocol so other agents can call it remotely.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
import uvicorn

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.console import logger


async def check_sub_agent_health(
    name: str,
    url: str,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Check health of a sub-agent by its URL."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try /health first, fall back to agent card if 404
            response = await client.get(f"{url}/health")
            if response.status_code == 404:
                response = await client.get(f"{url}/.well-known/agent-card.json")
            return {
                "name": name,
                "url": url,
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
            }
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "healthy": False,
            "error": str(e),
        }


def _add_custom_routes(app: Any, agent: Agent, persona: Persona) -> None:
    """Add custom routes to the Starlette app for health, status, and chat."""

    async def health_check(request: Request) -> JSONResponse:
        """Simple health check endpoint."""
        return JSONResponse({"status": "ok", "agent": persona.name})

    async def chat_stream(request: Request) -> Response:
        """Handle chat requests with SSE streaming response."""
        try:
            body = await request.json()
            message_text = body.get("message", "")
            if not message_text:
                return JSONResponse({"error": "No message provided"}, status_code=400)

            # Create the content for the agent
            content = types.Content(
                role="user",
                parts=[types.Part(text=message_text)],
            )

            async def event_generator() -> AsyncGenerator[str, None]:
                """Generate SSE events from agent response."""
                try:
                    async for event in agent.stream_chat_events_async(content):
                        event_type = event.get("type", "")
                        if event_type == "text":
                            yield f"data: {json.dumps(event)}\n\n"
                        elif event_type in ("tool_call", "tool_result"):
                            yield f"data: {json.dumps(event)}\n\n"
                        else:
                            # Pass through other event types
                            yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def agents_status(request: Request) -> JSONResponse:
        """Return the agent hierarchy with health status."""
        result: dict[str, Any] = {
            "root": {
                "name": persona.name,
                "type": persona.type,
                "healthy": True,
            },
            "sub_agents": {},
        }

        # If this persona has sub-agents, check their health
        if persona.sub_agents:
            tasks = []
            for sub_agent_name in persona.sub_agents:
                # Build URL for sub-agent (docker-compose service naming)
                service_name = sub_agent_name.lower().replace("_", "-")
                sub_agent_url = f"http://{service_name}:8001"
                tasks.append(check_sub_agent_health(sub_agent_name, sub_agent_url))

            # Check all sub-agents in parallel
            if tasks:
                statuses = await asyncio.gather(*tasks, return_exceptions=True)
                for status in statuses:
                    if isinstance(status, dict):
                        name = status.get("name", "unknown")
                        result["sub_agents"][name] = status
                    elif isinstance(status, Exception):
                        logger.warning(f"Error checking sub-agent: {status}")

        return JSONResponse(result)

    async def agent_details(request: Request) -> JSONResponse:
        """Get detailed information about a specific agent."""
        agent_name = request.query_params.get("name", "")

        # Check if requesting root agent details
        if not agent_name or agent_name.lower() == persona.name.lower() or agent_name.lower() == "root":
            return JSONResponse({
                "name": persona.name,
                "type": persona.type,
                "url": "self",
                "healthy": True,
                "description": persona.description,
                "model": persona.model,
                "activity": "idle",
                "sub_agents": persona.sub_agents,
            })

        # Check if it's a sub-agent
        if agent_name not in persona.sub_agents:
            return JSONResponse(
                {"error": f"Agent '{agent_name}' not found"},
                status_code=404,
            )

        # Get details from the sub-agent
        service_name = agent_name.lower().replace("_", "-")
        sub_agent_url = f"http://{service_name}:8001"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to get agent card for description
                agent_card = {}
                try:
                    card_resp = await client.get(f"{sub_agent_url}/.well-known/agent-card.json")
                    if card_resp.status_code == 200:
                        agent_card = card_resp.json()
                except Exception:
                    pass

                # Check health
                health_status = await check_sub_agent_health(agent_name, sub_agent_url)

                return JSONResponse({
                    "name": agent_name,
                    "type": agent_card.get("type", agent_name),
                    "url": sub_agent_url,
                    "healthy": health_status.get("healthy", False),
                    "error": health_status.get("error", ""),
                    "description": agent_card.get("description", ""),
                    "activity": "idle",
                })
        except Exception as e:
            return JSONResponse({
                "name": agent_name,
                "url": sub_agent_url,
                "healthy": False,
                "error": str(e),
                "activity": "unknown",
            })

    async def memory_status(request: Request) -> JSONResponse:
        """Get durable memory status for this agent instance."""
        return JSONResponse(agent.memory_status())

    async def memory_clear(request: Request) -> JSONResponse:
        """Clear durable memory (transcript + state) for this agent instance."""
        agent.clear_memory()
        return JSONResponse({"ok": True, **agent.memory_status()})

    # Insert routes at the BEGINNING of the routes list so they take precedence
    # (Starlette matches routes in order, and the A2A app may have a catch-all)
    app.routes.insert(0, Route("/health", health_check, methods=["GET"]))
    app.routes.insert(1, Route("/chat", chat_stream, methods=["POST"]))
    app.routes.insert(2, Route("/memory", memory_status, methods=["GET"]))
    app.routes.insert(3, Route("/memory/clear", memory_clear, methods=["POST"]))
    app.routes.insert(4, Route("/agents/status", agents_status, methods=["GET"]))
    app.routes.insert(5, Route("/agents/details", agent_details, methods=["GET"]))


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

    # Wrap agent with A2A and create Starlette app.
    a2a_app = to_a2a(agent.adk_agent, port=port)

    # Add custom endpoints for health, chat, and agent status
    _add_custom_routes(a2a_app, agent, persona)

    logger.info(f"Starting A2A server for {persona_path.name} on port {port}")
    logger.info(f"Agent card available at: http://0.0.0.0:{port}/.well-known/agent-card.json")
    logger.info(f"Health check at: http://0.0.0.0:{port}/health")
    logger.info(f"Agent status at: http://0.0.0.0:{port}/agents/status")

    # Run the server (blocks until shutdown).
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


def create_app_for_persona(persona_path: Path, *, port: int = 8001):
    """Create the Starlette app without running it (for testing/embedding).

    Args:
        persona_path: Path to persona YAML file.
        port: Port the server will run on.

    Returns:
        The configured Starlette application.
    """
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")

    persona = Persona.from_yaml(persona_path)
    agent = Agent(persona=persona)
    a2a_app = to_a2a(agent.adk_agent, port=port)

    # Add custom endpoints
    _add_custom_routes(a2a_app, agent, persona)

    return a2a_app


if __name__ == "__main__":
    main()
