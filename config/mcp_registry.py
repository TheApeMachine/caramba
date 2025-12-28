"""MCP server registry configuration.

We intentionally keep the Python module name different from the YAML manifest
(`config/mcp_servers.yml`) to avoid tooling/import resolution confusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


MCPTransport = Literal["streamable-http", "sse", "stdio"]


class MCPServerConfig(BaseModel):
    """One MCP server endpoint/transport definition."""

    # Human-readable name (optional; defaults to the mapping key in YAML).
    name: str | None = None
    transport: MCPTransport = "streamable-http"

    # HTTP/SSE transports
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    encoding: str | None = None


def load_mcp_servers(path: Path = Path("config/mcp_servers.yml")) -> dict[str, MCPServerConfig]:
    """Load MCP server registry from YAML.

    Returns:
        Mapping from tool name to MCPServerConfig.
    """
    if not path.exists():
        return {}

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(raw).__name__}")

    out: dict[str, MCPServerConfig] = {}
    for key, cfg in raw.items():
        if not isinstance(key, str):
            continue
        if not isinstance(cfg, dict):
            raise TypeError(f"Expected mapping for '{key}' in {path}, got {type(cfg).__name__}")
        out[key] = MCPServerConfig(**cfg, name=str(cfg.get("name") or key))
    return out

