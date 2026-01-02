"""Implements the Google Agent Development Kit

These make the agents compatible with the Agent-to-Agent protocol.
"""
from __future__ import annotations

import asyncio
import socket
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
import os
import re

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.agents.run_config import RunConfig, StreamingMode
import httpx

from caramba.ai.persona import Persona
from caramba.console import logger

# Silence known-noisy runtime warnings from upstream libs.
# (These are informational and overwhelm the REPL.)
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\] BaseAuthenticatedTool:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*non-text parts in the response.*",
)

@dataclass(frozen=True)
class _McpEndpoint:
    url: str
    transport: str | None = None  # e.g. "streamable-http" or "sse"
    headers: dict[str, str] | None = None


def _iter_persona_tool_names(tools: object) -> list[str]:
    """Normalize persona tools into a list of tool/server names."""
    if not tools:
        return []
    if isinstance(tools, list):
        out: list[str] = []
        for item in tools:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Some legacy persona YAMLs use: tools: - name: "math"
                name = item.get("name")
                if isinstance(name, str) and name:
                    out.append(name)
        return out
    return []


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _expand_env_placeholders(payload: object) -> object:
    """Recursively expand ${ENV_VAR} placeholders using os.environ.

    Missing env vars are left as-is so unused tools don't break startup.
    """
    if isinstance(payload, dict):
        return {k: _expand_env_placeholders(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_expand_env_placeholders(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_expand_env_placeholders(v) for v in payload)
    if not isinstance(payload, str):
        return payload

    if "${" not in payload:
        return payload

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        val = os.getenv(name)
        return match.group(0) if val is None else str(val)

    return _ENV_PATTERN.sub(_replace, payload)


def _load_mcp_endpoints() -> dict[str, _McpEndpoint]:
    """Load MCP endpoints from config files.

    Sources:
    - `config/mcp_servers.yml` (legacy consolidated mapping)
    - `config/tools/*.yml` (one tool per file)
    """
    merged: dict[str, _McpEndpoint] = {}

    # 1) Legacy consolidated config
    p = Path("config/mcp_servers.yml")
    if p.exists():
        try:
            import yaml  # used elsewhere in repo

            payload = _expand_env_placeholders(yaml.safe_load(p.read_text()) or {})
            if isinstance(payload, dict):
                for name, entry in payload.items():
                    if isinstance(entry, dict):
                        url = entry.get("url")
                        transport = entry.get("transport")
                        headers = entry.get("headers")
                        if isinstance(url, str) and url:
                            merged[str(name)] = _McpEndpoint(
                                url=url,
                                transport=str(transport) if isinstance(transport, str) and transport else None,
                                headers=dict(headers) if isinstance(headers, dict) else None,
                            )
        except Exception as e:
            logger.warning(f"Failed to parse {p}: {e}")

    # 2) Per-tool configs
    tools_dir = Path("config/tools")
    if tools_dir.exists():
        try:
            import yaml

            for yml in tools_dir.glob("*.yml"):
                try:
                    payload = _expand_env_placeholders(yaml.safe_load(yml.read_text()) or {})
                    if isinstance(payload, dict):
                        for name, entry in payload.items():
                            if isinstance(entry, dict):
                                url = entry.get("url")
                                transport = entry.get("transport")
                                headers = entry.get("headers")
                                if isinstance(url, str) and url:
                                    merged[str(name)] = _McpEndpoint(
                                        url=url,
                                        transport=str(transport) if isinstance(transport, str) and transport else None,
                                        headers=dict(headers) if isinstance(headers, dict) else None,
                                    )
                except Exception as e:
                    logger.warning(f"Failed to parse {yml}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load MCP tool configs: {e}")

    return merged


_MCP_UNHEALTHY_WARNED: set[str] = set()


def _url_is_reachable(url: str, *, timeout_sec: float = 0.25) -> bool:
    """Best-effort TCP reachability check to avoid opaque TaskGroup errors."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        if not host:
            return False
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except Exception:
        return False


def _endpoint_is_healthy(endpoint: _McpEndpoint, *, timeout_sec: float = 0.5) -> bool:
    """Best-effort health check to avoid hanging MCP session inits.

    For our own SSE servers we expose `/health`. For other servers, we fall back to
    basic TCP reachability.
    """
    if not _url_is_reachable(endpoint.url, timeout_sec=min(timeout_sec, 0.25)):
        return False

    if httpx is None:
        return True

    transport = (endpoint.transport or "").strip().lower()
    if transport == "sse":
        # For SSE servers, don't GET /sse (it streams). Use /health if available.
        try:
            parsed = urlparse(endpoint.url)
            if not parsed.scheme or not parsed.hostname:
                return True
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            health_url = f"{parsed.scheme}://{parsed.hostname}:{port}/health"
            r = httpx.get(health_url, timeout=timeout_sec)
            return 200 <= r.status_code < 300
        except Exception:
            return False

    if transport in {"streamable-http", "streamable_http", "streamablehttp"}:
        # Streamable HTTP MCP endpoints are typically mounted at `/mcp`.
        # Avoid hanging session initialization by verifying the endpoint responds quickly.
        try:
            r = httpx.get(
                endpoint.url,
                timeout=timeout_sec,
                headers={"accept": "application/json"},
                follow_redirects=False,
            )
            # Many servers respond 405 (GET not allowed). That's fine: it proves the route exists.
            if r.status_code == 404:
                return False
            if 200 <= r.status_code < 500:
                return True
            return False
        except Exception:
            return False

    # Unknown/unspecified transport: TCP reachability is already checked above.
    return True


class BestEffortMcpToolset(McpToolset):
    """McpToolset that degrades to "no tools" on failure.

    This must remain a `BaseToolset` instance for ADK validation, hence the subclass.
    """

    def __init__(self, *args: Any, label: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._label = label

    async def get_tools(self, readonly_context: Any):
        try:
            return await super().get_tools(readonly_context)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning(
                f"Disabling MCP toolset '{self._label}' due to error: {type(e).__name__}: {e}"
            )
            return []

    async def get_tools_with_prefix(self, ctx: Any):
        # Some ADK versions call this method instead of get_tools().
        try:
            return await super().get_tools_with_prefix(ctx)  # type: ignore[misc]
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning(
                f"Disabling MCP toolset '{self._label}' due to error: {type(e).__name__}: {e}"
            )
            return []


def _connection_params_for(endpoint: _McpEndpoint):
    """Create the correct ADK MCP connection params for an endpoint."""
    transport = (endpoint.transport or "").strip().lower()
    if transport in {"streamable-http", "streamable_http", "streamablehttp"}:
        if StreamableHTTPConnectionParams is None:
            raise RuntimeError(
                "MCP server is configured with transport=streamable-http, "
                "but this ADK install does not expose StreamableHTTPConnectionParams. "
                "Upgrade google-adk to a version that supports streamable-http MCP, or adjust MCP transport to SSE."
            )
        return StreamableHTTPConnectionParams(  # type: ignore[misc]
            url=endpoint.url,
            headers=endpoint.headers or {},
        )
    if transport in {"sse", "server-sent-events", "server_sent_events"}:
        return SseConnectionParams(url=endpoint.url, headers=endpoint.headers or {})  # type: ignore[misc]

    # Unknown/unspecified: prefer streamable-http when available, else SSE.
    if StreamableHTTPConnectionParams is not None:
        return StreamableHTTPConnectionParams(  # type: ignore[misc]
            url=endpoint.url,
            headers=endpoint.headers or {},
        )
    return SseConnectionParams(url=endpoint.url, headers=endpoint.headers or {})  # type: ignore[misc]


def _make_adk_model(persona: Persona):
    """Choose an ADK model implementation with safe fallbacks.

    ADK warns when using Gemini via LiteLLM; use native Gemini if available.
    """
    model_name = str(persona.model or "")
    if model_name.startswith("gemini/"):
        gemini_model = model_name.split("/", 1)[1]
        # Try a few known ADK import locations for Gemini, then fall back to LiteLLM.
        try:
            from google.adk.models import Gemini  # type: ignore

            return Gemini(model=gemini_model)
        except Exception:
            try:
                from google.adk.models.gemini import Gemini  # type: ignore

                return Gemini(model=gemini_model)
            except Exception:
                return LiteLlm(model=model_name)
    return LiteLlm(model=model_name)


class Agent:
    """Agent is a flexible wrapper to build AI agents."""
    def __init__(
        self,
        persona: Persona,
        *,
        app_name: str = "caramba",
        user_id: str = "user",
        session_service: InMemorySessionService | None = None,
        session_id: str | None = None,
        # Optional override: single MCP URL (primarily for tests / quick wiring).
        mcp_url: str | None = None,
    ):
        self.persona = persona
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id or str(uuid4())

        # MCP tool wiring:
        # - Persona declares tool server names in `persona.tools`
        # - We resolve those names to URLs via config and create one toolset per server
        tools: list[Any] = []
        tool_names = _iter_persona_tool_names(getattr(persona, "tools", None))

        endpoints = _load_mcp_endpoints()

        # If an explicit URL is provided, attach it as a single toolset (no name resolution).
        if isinstance(mcp_url, str) and mcp_url:
            ep = _McpEndpoint(url=mcp_url)
            if _endpoint_is_healthy(ep):
                tools.append(
                    BestEffortMcpToolset(
                        connection_params=_connection_params_for(ep),
                        label=mcp_url,
                    )
                )
            else:
                logger.warning(f"MCP unreachable at {mcp_url}; continuing without MCP tools.")
        else:
            for name in tool_names:
                endpoint = endpoints.get(name)
                if not endpoint:
                    logger.warning(f"Unknown MCP tool/server '{name}' (no URL found in config).")
                    continue
                if not _endpoint_is_healthy(endpoint):
                    # Avoid spamming the REPL: many agents share the same tool list.
                    key = f"{name}@{endpoint.url}"
                    if key not in _MCP_UNHEALTHY_WARNED:
                        _MCP_UNHEALTHY_WARNED.add(key)
                        logger.warning(
                            f"MCP tool/server '{name}' unhealthy/unreachable at {endpoint.url}; skipping."
                        )
                    continue
                tools.append(
                    BestEffortMcpToolset(
                        connection_params=_connection_params_for(endpoint),
                        label=name,
                    )
                )

        self.adk_agent = LlmAgent(
            model=_make_adk_model(persona),
            name=persona.name,
            description=persona.description,
            instruction=persona.instructions,
            tools=tools
        )
        self.session_service = session_service or InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        self.session_ready = False

    def reset_session(self, *, session_id: str | None = None) -> None:
        """Reset the ADK session for a fresh run (useful for external transcripts)."""
        self.session_id = session_id or str(uuid4())
        self.session_ready = False

    def run(self, input: str) -> str:
        """Run a single turn and return the final text output."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(input))
        raise RuntimeError("Agent.run() cannot be called from an active event loop; use `await Agent.run_async(...)`.")

    async def run_async(self, input: str) -> str:
        """Async variant of `run()`."""
        text_chunks: list[str] = []
        async for event in self.run_events_async(input):
            ev_content = getattr(event, "content", None)
            parts = getattr(ev_content, "parts", None)
            if not parts:
                continue
            for part in parts:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt:
                    text_chunks.append(txt)

        return "".join(text_chunks).strip()

    async def run_events_async(self, input: str, *, run_config: Any | None = None):
        """Yield ADK runtime events for a single user turn."""
        await self._ensure_session()
        content = types.Content(role="user", parts=[types.Part(text=str(input))])
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content,
                run_config=run_config,
            ):
                yield event
        except BaseExceptionGroup as eg:  # py>=3.11: unwrap TaskGroup failures
            lines = ["Unhandled exception group while running agent:"]
            for i, exc in enumerate(getattr(eg, "exceptions", []) or [], start=1):
                lines.append(f"  ({i}) {type(exc).__name__}: {exc}")
            raise RuntimeError("\n".join(lines)) from eg
        except Exception as e:
            # Improve the most common failure mode: MCP session creation errors are opaque.
            msg = str(e)
            if "Failed to create MCP session" in msg:
                endpoints = _load_mcp_endpoints()
                tool_names = _iter_persona_tool_names(getattr(self.persona, "tools", None))
                details = []
                for name in tool_names:
                    ep = endpoints.get(name)
                    if ep is None:
                        details.append(f"  - {name}: (no config url)")
                    else:
                        details.append(f"  - {name}: url={ep.url} transport={ep.transport or 'auto'}")
                raise RuntimeError(
                    "Failed to create MCP session.\n"
                    "Configured tool servers:\n"
                    + "\n".join(details)
                    + "\n\n"
                    "This usually indicates a transport mismatch (streamable-http vs SSE) "
                    "or an MCP server that is up but not speaking MCP on the configured endpoint."
                ) from e
            raise

    async def stream_text_async(
        self,
        input: str,
        *,
        streaming_mode: str | None = "sse",
        temperature: float | None = None,
        run_config: Any | None = None,
    ):
        """Yield model text incrementally as it's produced.

        Note: Whether you get true token streaming depends on:
        - ADK RunConfig streaming settings
        - The underlying model/provider support for streaming
        """
        if run_config is None and RunConfig is not None and streaming_mode and StreamingMode is not None:
            mode_key = str(streaming_mode).strip().lower()
            mode = {
                "none": StreamingMode.NONE,
                "sse": StreamingMode.SSE,
                "bidi": getattr(StreamingMode, "BIDI", StreamingMode.SSE),
            }.get(mode_key, StreamingMode.SSE)

            if temperature is None:
                temperature = float(getattr(self.persona, "temperature", 0.0))

            run_config = RunConfig(streaming_mode=mode, temperature=float(temperature))  # type: ignore[misc]

        buffer = ""
        async for event in self.run_events_async(input, run_config=run_config):
            ev_content = getattr(event, "content", None)
            parts = getattr(ev_content, "parts", None)
            if not parts:
                continue
            for part in parts:
                txt = getattr(part, "text", None)
                if not isinstance(txt, str) or not txt:
                    continue
                # ADK/providers differ: sometimes this is a delta, sometimes it's the full-so-far text.
                # Emit only the incremental suffix when it looks like "full so far".
                if txt.startswith(buffer):
                    delta = txt[len(buffer) :]
                    if delta:
                        buffer = txt
                        yield delta
                    continue
                if buffer.startswith(txt):
                    # Older chunk repeated; ignore.
                    continue

                buffer += txt
                yield txt

    async def stream_chat_events_async(
        self,
        input: str,
        *,
        streaming_mode: str | None = "sse",
        temperature: float | None = None,
        run_config: Any | None = None,
    ):
        """Yield a structured stream of text/tool events.

        Emits dicts like:
        - {"type": "text", "text": "..."}  (incremental)
        - {"type": "tool_call", "name": "...", "args": {...}, "id": "..."}
        - {"type": "tool_result", "name": "...", "response": {...}, "id": "..."}
        """
        # Mirror `stream_text_async` run_config behavior.
        if run_config is None and RunConfig is not None and streaming_mode and StreamingMode is not None:
            mode_key = str(streaming_mode).strip().lower()
            mode = {
                "none": StreamingMode.NONE,
                "sse": StreamingMode.SSE,
                "bidi": getattr(StreamingMode, "BIDI", StreamingMode.SSE),
            }.get(mode_key, StreamingMode.SSE)

            if temperature is None:
                temperature = float(getattr(self.persona, "temperature", 0.0))

            run_config = RunConfig(streaming_mode=mode, temperature=float(temperature))  # type: ignore[misc]

        buffer = ""
        seen_tool_calls: set[str] = set()
        seen_tool_results: set[str] = set()

        async for event in self.run_events_async(input, run_config=run_config):
            ev_content = getattr(event, "content", None)
            parts = getattr(ev_content, "parts", None)
            if not parts:
                continue

            for part in parts:
                # Tool call
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    fc_id = str(getattr(fc, "id", "") or "")
                    if fc_id and fc_id in seen_tool_calls:
                        pass
                    else:
                        if fc_id:
                            seen_tool_calls.add(fc_id)
                        yield {
                            "type": "tool_call",
                            "id": fc_id,
                            "name": getattr(fc, "name", None),
                            "args": getattr(fc, "args", None) or getattr(fc, "partial_args", None),
                        }

                # Tool result
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    fr_id = str(getattr(fr, "id", "") or "")
                    if fr_id and fr_id in seen_tool_results:
                        pass
                    else:
                        if fr_id:
                            seen_tool_results.add(fr_id)
                        yield {
                            "type": "tool_result",
                            "id": fr_id,
                            "name": getattr(fr, "name", None),
                            "response": getattr(fr, "response", None),
                        }

                # Text (delta-ish)
                txt = getattr(part, "text", None)
                if not isinstance(txt, str) or not txt:
                    continue
                if txt.startswith(buffer):
                    delta = txt[len(buffer) :]
                    if delta:
                        buffer = txt
                        yield {"type": "text", "text": delta}
                    continue
                if buffer.startswith(txt):
                    continue

                buffer += txt
                yield {"type": "text", "text": txt}

    async def _ensure_session(self) -> Session | None:
        if self.session_ready:
            return
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self.session_ready = True
        return await self.session_service.get_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )