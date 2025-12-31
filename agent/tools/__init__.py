"""Agent tools via MCP servers.

The agent system should prefer **remote MCP servers** (typically Docker-managed,
see `docker-compose.yml`) and fall back to spawning local stdio servers only when
explicitly configured.

Tool wiring is manifest-driven via `config/mcp_servers.yml`.
"""
from __future__ import annotations

from typing import Any, cast

from agents.mcp import (
    MCPServer,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
)

from caramba.config.mcp_registry import MCPServerConfig, load_mcp_servers


_MCP_REGISTRY_CACHE: dict[str, MCPServerConfig] | None = None


def _get_mcp_registry(*, reload: bool = False) -> dict[str, MCPServerConfig]:
    global _MCP_REGISTRY_CACHE
    if reload or _MCP_REGISTRY_CACHE is None:
        _MCP_REGISTRY_CACHE = load_mcp_servers()
    return _MCP_REGISTRY_CACHE


class Tool:
    """Generic tool class."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self) -> MCPServer:
        """Get the MCP client for the tool.

        Returns:
            MCP client for the tool.
        """
        return self.mcp_client()

    def mcp_client(self) -> MCPServer:
        """Create an MCP client for this tool based on the registry manifest."""
        registry = _get_mcp_registry()
        if self.name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(
                f"Unknown MCP server '{self.name}'. "
                f"Define it in config/mcp_servers.yml. Available: [{available}]"
            )

        cfg = registry[self.name]

        if cfg.transport == "streamable-http":
            if not cfg.url:
                raise ValueError(f"MCP server '{self.name}' missing 'url' for streamable-http.")
            return MCPServerStreamableHttp(
                cache_tools_list=True,
                name=cfg.name or self.name,
                params={
                    "url": cfg.url,
                    "headers": cfg.headers or {},
                },
            )

        if cfg.transport == "sse":
            if not cfg.url:
                raise ValueError(f"MCP server '{self.name}' missing 'url' for sse.")
            return MCPServerSse(
                cache_tools_list=True,
                name=cfg.name or self.name,
                params={
                    "url": cfg.url,
                    "headers": cfg.headers or {},
                },
            )

        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"MCP server '{self.name}' missing 'command' for stdio.")
            params = cast(
                MCPServerStdioParams,
                {
                    "command": cfg.command,
                    "args": cfg.args or [],
                    "env": cfg.env or {},
                },
            )
            if cfg.cwd is not None:
                params["cwd"] = cfg.cwd
            if cfg.encoding is not None:
                params["encoding"] = cfg.encoding
            return MCPServerStdio(
                cache_tools_list=True,
                name=cfg.name or self.name,
                params=params,
            )

        raise ValueError(f"Unknown transport '{cfg.transport}' for MCP server '{self.name}'.")


__all__ = ["Tool"]