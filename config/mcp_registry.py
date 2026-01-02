"""MCP server registry configuration.

We intentionally keep the Python module name different from the YAML manifest
(`config/mcp_servers.yml`) to avoid tooling/import resolution confusion.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import warnings
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


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _expand_env_placeholders(payload: object) -> object:
    """Recursively expand ${ENV_VAR} placeholders using os.environ.

    Notes:
    - Missing env vars are left as-is (so unused tools don't break registry load).
    - Only string values are interpolated; keys are not modified.
    """
    if isinstance(payload, Mapping):
        return {k: _expand_env_placeholders(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_expand_env_placeholders(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_expand_env_placeholders(v) for v in payload)
    if not isinstance(payload, str):
        return payload

    matches = list(_ENV_PATTERN.finditer(payload))
    if not matches:
        return payload

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        val = os.getenv(name)
        # If unset, keep the placeholder so registry load stays non-fatal.
        return payload[match.start() : match.end()] if val is None else str(val)

    # If the entire string is exactly "${VAR}", preserve non-string types? Not needed here.
    return _ENV_PATTERN.sub(_replace, payload)


def load_mcp_servers(path: Path = Path("config/mcp_servers.yml")) -> dict[str, MCPServerConfig]:
    """Load MCP server registry from YAML.

    Returns:
        Mapping from tool name to MCPServerConfig.
    """
    if not path.exists():
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(raw).__name__}")

    expanded = _expand_env_placeholders(raw)
    if not isinstance(expanded, dict):
        raise TypeError(f"Expected mapping at {path} after env expansion, got {type(expanded).__name__}")
    raw = expanded

    out: dict[str, MCPServerConfig] = {}
    for key, cfg in raw.items():
        if not isinstance(key, str):
            warnings.warn(
                f"Ignoring non-string MCP server key {key!r} in {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not isinstance(cfg, dict):
            raise TypeError(f"Expected mapping for '{key}' in {path}, got {type(cfg).__name__}")
        cfg_dict = dict(cfg)
        explicit_name = cfg_dict.pop("name", None)
        out[key] = MCPServerConfig(**cfg_dict, name=str(explicit_name or key))
    return out

