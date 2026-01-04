"""Implements the Google Agent Development Kit

These make the agents compatible with the Agent-to-Agent protocol.
"""
from __future__ import annotations

import asyncio
import random
import re
import warnings
from typing import Any
from uuid import uuid4

from google.adk.agents import LlmAgent
try:
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
except ImportError:
    # Fallback: try alternative import path
    try:
        from google.adk.a2a import RemoteA2aAgent  # type: ignore[import-untyped]
    except ImportError:
        RemoteA2aAgent = None  # type: ignore[assignment,misc]

# Standard A2A agent card well-known path
AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"

from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models import Gemini
from jsonschema_pydantic import jsonschema_to_pydantic

from caramba.ai.persona import Persona
from caramba.console import logger
from caramba.ai.mcp import (
    BestEffortMcpToolset,
    McpEndpoint,
    connection_params_for,
    endpoint_is_healthy,
    iter_persona_tool_names,
    load_mcp_endpoints,
)

# Silence known-noisy runtime warnings from upstream libs.
# (These are informational and overwhelm the REPL.)
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\] BaseAuthenticatedTool:.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")


MCP_UNHEALTHY_WARNED: set[str] = set()

_VALID_NAME_RE = re.compile(r"[^0-9A-Za-z_]")


def _to_valid_identifier(name: str, *, fallback: str = "agent") -> str:
    """Convert an arbitrary display name into an ADK-valid identifier."""
    s = (name or "").strip()
    if not s:
        return fallback
    s = s.replace("-", "_").replace(" ", "_")
    s = _VALID_NAME_RE.sub("_", s)
    # Must start with a letter or underscore.
    if not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    # Collapse repeated underscores and trim.
    s = re.sub(r"_+", "_", s).strip("_") or fallback
    return s

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
        # - Persona declares tool server names in `persona.tools` or `persona.mcp_servers` (legacy)
        # - We resolve those names to URLs via config and create one toolset per server
        tools: list[Any] = []
        # Support both `tools` and legacy `mcp_servers` fields.
        tool_names = iter_persona_tool_names(
            getattr(persona, "tools", None) or getattr(persona, "mcp_servers", None)
        )

        endpoints = load_mcp_endpoints()

        # If an explicit URL is provided, attach it as a single toolset (no name resolution).
        if isinstance(mcp_url, str) and mcp_url:
            ep = McpEndpoint(url=mcp_url)
            if endpoint_is_healthy(ep):
                tools.append(
                    BestEffortMcpToolset(
                        connection_params=connection_params_for(ep),
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
                if not endpoint_is_healthy(endpoint):
                    # Avoid spamming the REPL: many agents share the same tool list.
                    key = f"{name}@{endpoint.url}"
                    if key not in MCP_UNHEALTHY_WARNED:
                        MCP_UNHEALTHY_WARNED.add(key)
                        logger.warning(
                            f"MCP tool/server '{name}' unhealthy/unreachable at {endpoint.url}; skipping."
                        )
                    continue
                tools.append(
                    BestEffortMcpToolset(
                        connection_params=connection_params_for(endpoint),
                        label=name,
                    )
                )

        # --- Model selection / normalization ---
        #
        # If persona.model is empty or "random", select one provider at startup.
        #
        # NOTE: ADK's `LlmAgent` validates `model` as either a string or a BaseLlm
        # instance. A lightweight "proxy" object (like our previous RandomModel)
        # is rejected by Pydantic, so we choose a concrete ADK model here.
        # Otherwise, support a few common model string conventions:
        # - "gemini/<model>" (LiteLLM-style)
        # - "google/<model>" (internal convention)
        # - any other string -> routed to LiteLLM
        #
        # ADK's native Gemini integration expects the bare Gemini model name (e.g. "gemini-3-pro-preview"),
        # not a prefixed identifier like "gemini/..." or "google/...".
        from caramba.ai.models.random_model import RandomModel

        raw_model = (persona.model or "").strip()
        if not raw_model or raw_model.lower() == "random":
            model_id = random.choice(RandomModel.MODEL_POOL)
            if model_id.startswith(("gemini/", "google/")):
                gemini_model = model_id.split("/", 1)[1].strip()
                self.model = Gemini(model=gemini_model)
            else:
                self.model = LiteLlm(model=model_id)
        elif raw_model.startswith(("gemini/", "google/")):
            gemini_model = raw_model.split("/", 1)[1].strip()
            self.model = Gemini(model=gemini_model)
        else:
            self.model = LiteLlm(model=raw_model)

        # Build sub-agents (RemoteA2aAgent instances) if persona declares them.
        sub_agents: list[Any] = []
        if persona.sub_agents:
            # Resolve sub-agent names to A2A agent card URLs.
            # In docker-compose, each persona service is accessible via its service name.
            # The agent card is at: http://<service_name>:8001/.well-known/agent-card.json
            if RemoteA2aAgent is not None:
                for sub_agent_name in persona.sub_agents:
                    # Use docker-compose service name (persona name, lowercase, hyphens for underscores).
                    service_name = sub_agent_name.lower().replace("_", "-")
                    agent_card_url = f"http://{service_name}:8001{AGENT_CARD_WELL_KNOWN_PATH}"
                    sub_agents.append(
                        RemoteA2aAgent(  # type: ignore[misc]
                            name=sub_agent_name,
                            description=f"Expert agent: {sub_agent_name}",
                            agent_card=agent_card_url,
                        )
                    )

        # ADK's LlmAgent expects a list; passing None triggers Pydantic validation errors.
        sub_agents_list = list(sub_agents) if sub_agents else []

        # ADK validates agent names as identifiers; use persona.type when available.
        agent_internal_name = _to_valid_identifier(persona.type or persona.name, fallback="caramba_agent")

        self.adk_agent = LlmAgent(
            model=self.model,  # type: ignore[arg-type]  # RandomModel is compatible via __getattr__
            name=agent_internal_name,
            description=persona.description,
            instruction=persona.instructions,
            tools=tools,
            sub_agents=sub_agents_list,  # type: ignore[arg-type]  # RemoteA2aAgent extends BaseAgent
        )

        if persona.output_schema:
            self.adk_agent.output_schema = jsonschema_to_pydantic(persona.output_schema)

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

    def run(self, message: types.Content) -> str:
        """Run a single turn and return the final text output."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(message))

        raise RuntimeError("Agent.run() cannot be called from an active event loop; use `await Agent.run_async(...)`.")

    async def run_async(self, message: types.Content) -> str:
        """Async variant of `run()`."""
        text_chunks: list[str] = []
        async for event in self.run_events_async(message):
            ev_content = getattr(event, "content", None)
            parts = getattr(ev_content, "parts", None)
            if not parts:
                continue
            for part in parts:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt:
                    text_chunks.append(txt)

        return "".join(text_chunks).strip()

    async def run_events_async(self, message: types.Content):
        """Yield ADK runtime events for a single user turn."""
        # If the selected model supports resetting between requests, do it.
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

        await self._ensure_session()
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
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
                endpoints = load_mcp_endpoints()
                tool_names = iter_persona_tool_names(getattr(self.persona, "tools", None))
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

    # async def stream_text_async(
    #     self,
    # ):
    #     """Yield model text incrementally (deltas only)."""
    #     seen_text = ""
    #     async for event in self.run_events_async(message):
    #         ev_content = getattr(event, "content", None)
    #         parts = getattr(ev_content, "parts", None)
    #         if not parts:
    #             continue
    #         for part in parts:
    #             txt = getattr(part, "text", None)
    #             if isinstance(txt, str) and txt:
    #                 # ADK may send cumulative text - extract delta
    #                 if txt.startswith(seen_text):
    #                     delta = txt[len(seen_text):]
    #                     if delta:
    #                         seen_text = txt
    #                         yield delta
    #                 elif not seen_text:
    #                     seen_text = txt
    #                     yield txt

    async def stream_chat_events_async(
        self,
        message: types.Content,
    ):
        """Yield a structured stream of text/tool events.

        Emits dicts like:
        - {"type": "text", "text": "..."} (deltas only)
        - {"type": "tool_call", "name": "...", "args": {...}, "id": "..."}
        - {"type": "tool_result", "name": "...", "response": {...}, "id": "..."}
        """
        seen_text = ""
        try:
            async for event in self.run_events_async(message):
                ev_content = getattr(event, "content", None)
                parts = getattr(ev_content, "parts", None)
                if not parts:
                    continue

                for part in parts:
                    # Tool call
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        yield {
                            "type": "tool_call",
                            "id": str(getattr(fc, "id", "") or ""),
                            "name": getattr(fc, "name", None),
                            "args": getattr(fc, "args", None) or getattr(fc, "partial_args", None),
                        }

                    # Tool result
                    fr = getattr(part, "function_response", None)
                    if fr is not None:
                        yield {
                            "type": "tool_result",
                            "id": str(getattr(fr, "id", "") or ""),
                            "name": getattr(fr, "name", None),
                            "response": getattr(fr, "response", None),
                        }

                    # Text - extract delta from cumulative ADK output
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt:
                        # ADK/provider variance:
                        # - some providers stream *cumulative* text (txt grows each event)
                        # - others stream *delta* chunks (txt is just the new slice)
                        #
                        # We unify both cases by emitting ONLY the new suffix that
                        # wasn't already streamed (prevents end-of-stream duplication).
                        if not seen_text:
                            seen_text = txt
                            yield {"type": "text", "text": txt}
                            continue

                        # Case 1: cumulative replay (txt contains everything so far).
                        if txt.startswith(seen_text):
                            delta = txt[len(seen_text) :]
                            if delta:
                                seen_text = txt
                                yield {"type": "text", "text": delta}
                            continue

                        # Case 2: shorter prefix replay (can happen with retries).
                        if seen_text.startswith(txt):
                            continue

                        # Case 3: delta chunk (or overlapping resend). Compute the maximum
                        # overlap between the end of what we've seen and the start of txt,
                        # then only emit the non-overlapping suffix.
                        max_check = min(len(seen_text), len(txt), 2048)
                        overlap = 0
                        for k in range(max_check, 0, -1):
                            if seen_text.endswith(txt[:k]):
                                overlap = k
                                break

                        delta = txt[overlap:]
                        if not delta:
                            continue
                        seen_text = seen_text + delta
                        yield {"type": "text", "text": delta}
        except asyncio.CancelledError:
            # Propagate cancellations cleanly to callers.
            raise
        except Exception as e:
            # Do not let provider/session exceptions crash the REPL loop. Emit a readable
            # error message as part of the assistant stream and end the generator cleanly.
            provider = (self.persona.name or "agent").strip()
            err = str(e).strip() or repr(e)
            yield {
                "type": "text",
                "text": f"\n\n---\n\n**[{provider} error]** {err}\n",
            }
            return

    async def _ensure_session(self) -> Session:
        if self.session_ready:
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
            )

            if session is None:
                raise RuntimeError(f"Session {self.session_id} not found after session_ready=True")

            return session

        self.session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        if self.session is None:
            raise RuntimeError(f"Failed to create session {self.session_id}")

        self.session_ready = True
        return self.session