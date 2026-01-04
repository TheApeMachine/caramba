"""MCP wiring for Caramba agents.

This module isolates MCP endpoint configuration, health checks, and toolset wiring
from `ai/agent.py` to keep files small and responsibilities separated.

It supports both transports used in the project:
- `sse` for local tool servers exposing `/sse` + `/health`
- `streamable-http` for MCP servers exposing `/mcp` (session-based)
"""

from __future__ import annotations

import socket
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

from caramba.console import logger


@dataclass(frozen=True)
class McpEndpoint:
    """A single MCP endpoint.

    `transport` is one of:
    - "streamable-http"
    - "sse"
    """

    url: str
    transport: str | None = None
    headers: dict[str, str] | None = None


class BestEffortMcpToolset(McpToolset):
    """MCP toolset that returns no tools on failure.

    Used to keep REPL processes robust when optional MCP servers are down.
    """

    def __init__(self, *args: Any, label: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label = label

    async def get_tools(self, readonly_context: Any):
        try:
            return await super().get_tools(readonly_context)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning(
                f"Disabling MCP toolset '{self.label}' due to error: {type(e).__name__}: {e}"
            )
            return []

    async def get_tools_with_prefix(self, ctx: Any):
        try:
            return await super().get_tools_with_prefix(ctx)  # type: ignore[misc]
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning(
                f"Disabling MCP toolset '{self.label}' due to error: {type(e).__name__}: {e}"
            )
            return []


def iter_persona_tool_names(tools: object) -> list[str]:
    """Normalize persona tools into a list of tool/server names."""
    if not tools:
        return []
    if isinstance(tools, list):
        out: list[str] = []
        for item in tools:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name:
                    out.append(name)
            else:
                # Support pydantic models (e.g. ToolRef) and other objects with a `name` attribute.
                name = getattr(item, "name", None)
                if isinstance(name, str) and name:
                    out.append(name)
        return out
    return []


ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}")


def expand_env_placeholders(payload: object) -> object:
    """Expand ${ENV_VAR} placeholders using os.environ."""
    if isinstance(payload, dict):
        return {k: expand_env_placeholders(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [expand_env_placeholders(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(expand_env_placeholders(v) for v in payload)
    if not isinstance(payload, str):
        return payload
    if "${" not in payload:
        return payload

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        val = os.getenv(name)
        if val is None:
            if isinstance(default, str):
                return default
            return match.group(0)
        return str(val)

    return ENV_PATTERN.sub(replace, payload)


def parse_mcp_entry(entry: object) -> McpEndpoint | None:
    """Parse a YAML entry into an endpoint."""
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    transport = entry.get("transport")
    headers = entry.get("headers")
    if not isinstance(url, str) or not url:
        return None
    return McpEndpoint(
        url=url,
        transport=str(transport) if isinstance(transport, str) and transport else None,
        headers=dict(headers) if isinstance(headers, dict) else None,
    )


def load_mcp_endpoints() -> dict[str, McpEndpoint]:
    """Load MCP endpoints from config files."""
    merged: dict[str, McpEndpoint] = {}

    consolidated = Path("config/mcp_servers.yml")
    if consolidated.exists():
        payload = expand_env_placeholders(yaml.safe_load(consolidated.read_text()) or {})
        if isinstance(payload, dict):
            for name, entry in payload.items():
                endpoint = parse_mcp_entry(entry)
                if endpoint is not None:
                    merged[str(name)] = endpoint

    tools_dir = Path("config/tools")
    if tools_dir.exists():
        for yml_path in tools_dir.glob("*.yml"):
            payload = expand_env_placeholders(yaml.safe_load(yml_path.read_text()) or {})
            if isinstance(payload, dict):
                for name, entry in payload.items():
                    endpoint = parse_mcp_entry(entry)
                    if endpoint is not None:
                        merged[str(name)] = endpoint

    return merged


def url_is_reachable(url: str, *, timeout_sec: float = 0.25) -> bool:
    """TCP reachability check (fast and protocol-agnostic)."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if not host:
        return False
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        # ConnectionRefusedError, timeout, DNS errors, etc.
        return False


def endpoint_is_healthy(endpoint: McpEndpoint, *, timeout_sec: float = 0.5) -> bool:
    """Health check that avoids protocol negotiation endpoints."""
    if not url_is_reachable(endpoint.url, timeout_sec=min(timeout_sec, 0.25)):
        return False

    transport = (endpoint.transport or "").strip().lower()
    if transport == "sse":
        parsed = urlparse(endpoint.url)
        if not parsed.scheme or not parsed.hostname:
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        health_url = f"{parsed.scheme}://{parsed.hostname}:{port}/health"
        try:
            r = httpx.get(health_url, timeout=timeout_sec)
            return 200 <= r.status_code < 300
        except httpx.HTTPError:
            return False

    if transport in {"streamable-http", "streamable_http", "streamablehttp"}:
        # Session-based protocol; probing with a raw GET triggers "Missing session ID".
        return True

    return True


def connection_params_for(endpoint: McpEndpoint):
    """Create ADK MCP connection params for an endpoint."""
    transport = (endpoint.transport or "").strip().lower()
    if transport in {"streamable-http", "streamable_http", "streamablehttp"}:
        headers = dict(endpoint.headers or {})
        if "accept" not in {k.lower() for k in headers.keys()}:
            headers["accept"] = "text/event-stream"
        return StreamableHTTPConnectionParams(url=endpoint.url, headers=headers)
    return SseConnectionParams(url=endpoint.url, headers=endpoint.headers or {})

