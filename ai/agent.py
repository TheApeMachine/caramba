"""Agent compatible with the Google Agent Development Kit

An agent is a wrapper around the Google Agent Development Kit, using
Model Context Protocol (MCP) tools.
It functions as an AI enabled automation tool and can be used for many
advanced use cases.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import warnings
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timezone

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models import Gemini
from jsonschema_pydantic import jsonschema_to_pydantic
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import Request

from caramba.ai.persona import Persona
from caramba.console import logger
from caramba.ai.process.transcript_store import TranscriptStore
from caramba.ai.mcp import (
    BestEffortMcpToolset,
    McpEndpoint,
    connection_params_for,
    endpoint_is_healthy,
    iter_persona_tool_names,
    load_mcp_endpoints,
)


warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\] BaseAuthenticatedTool:.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")

AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
MCP_UNHEALTHY_WARNED: set[str] = set()
VALID_NAME_RE = re.compile(r"[^0-9A-Za-z_]")


class Agent:
    """Agent is a flexible wrapper to build AI agents using MCP tools.

    Standard implementation uses agents for things like AI assisted research flows,
    and platform self-improvement.
    """
    def __init__(
        self,
        persona: Persona,
        *,
        app_name: str = "caramba",
        user_id: str = "user",
        session_service: InMemorySessionService | None = None,
        session_id: str | None = None,
        mcp_url: str | None = None,
    ):
        self.persona = persona
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id or str(uuid4())

        tools: list[Any] = []
        tool_names = iter_persona_tool_names(
            getattr(persona, "tools", None) or getattr(persona, "mcp_servers", None)
        )

        endpoints = load_mcp_endpoints()

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

        # Build expert agents as AgentTools for DELEGATION (not handoff).
        #
        # Key distinction in ADK:
        # - sub_agents + transfer_to_agent = HANDOFF: control moves entirely to sub-agent
        # - AgentTool in tools list = DELEGATION: root agent invokes sub-agent as a tool,
        #   maintains control, and synthesizes responses
        #
        # For the root orchestrator pattern (user always talks to root), we need delegation.
        expert_agent_tools: list[Any] = []
        if persona.sub_agents:
            # Resolve sub-agent names to A2A agent card URLs.
            # In docker-compose, each persona service is accessible via its service name.
            # The agent card is at: http://<service_name>:8001/.well-known/agent-card.json
            if RemoteA2aAgent is not None:
                for sub_agent_name in persona.sub_agents:
                    # Use docker-compose service name (persona name, lowercase, hyphens for underscores).
                    service_name = sub_agent_name.lower().replace("_", "-")
                    agent_card_url = f"http://{service_name}:8001{AGENT_CARD_WELL_KNOWN_PATH}"
                    remote_agent = RemoteA2aAgent(  # type: ignore[misc]
                        name=sub_agent_name,
                        description=f"Expert agent: {sub_agent_name}. Consult this agent for specialized knowledge.",
                        agent_card=agent_card_url,
                    )
                    # Wrap as AgentTool for delegation pattern (root maintains control)
                    expert_agent_tools.append(AgentTool(agent=remote_agent))

        # Combine MCP tools with expert agent tools
        all_tools = tools + expert_agent_tools

        # ADK validates agent names as identifiers; use persona.type when available.
        agent_internal_name = _to_valid_identifier(persona.type or persona.name, fallback="caramba_agent")

        self.adk_agent = LlmAgent(
            model=self.model,  # type: ignore[arg-type]  # RandomModel is compatible via __getattr__
            name=agent_internal_name,
            description=persona.description,
            instruction=persona.instructions,
            tools=all_tools,  # MCP tools + expert AgentTools for delegation
            # NOTE: We intentionally do NOT use sub_agents here.
            # sub_agents enables transfer_to_agent (handoff) which moves control away from root.
            # Instead, we use AgentTool wrapping for delegation (root maintains control).
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

        # Durable local transcript (lightweight "memory") for resuming work across restarts.
        # This is separate from ADK's in-memory session store and is intended to survive
        # docker restarts via the repo volume mount.
        self._transcript: TranscriptStore | None = None
        self._memory_state_path: Path | None = None
        try:
            repo_root = Path(__file__).resolve().parent.parent
            stable = _to_valid_identifier(persona.type or persona.name, fallback="agent").lower()
            transcript_path = repo_root / "artifacts" / "ai" / "chat" / f"{stable}.jsonl"
            self._memory_state_path = repo_root / "artifacts" / "ai" / "state" / f"{stable}.json"
            self._transcript = TranscriptStore(
                path=str(transcript_path),
                max_items=60,
                max_tokens=24_000,
                max_event_tokens=4_096,
                compact_after_bytes=2_000_000,
            )
            self._transcript.load()
        except Exception as e:
            # Memory must never prevent the agent from running.
            logger.warning(f"Transcript memory disabled due to error: {type(e).__name__}: {e}")
            self._transcript = None
            self._memory_state_path = None

    def server(self) -> Starlette:
        """Return the agent card for this agent."""
        app = to_a2a(self.adk_agent)
        app.routes.insert(0, Route("/health", self.health_check, methods=["GET"]))
        app.routes.insert(1, Route("/chat", self.chat_stream, methods=["POST"]))
        app.routes.insert(2, Route("/memory", self.memory_status, methods=["GET"]))
        app.routes.insert(3, Route("/memory/clear", self.memory_clear, methods=["POST"]))
        app.routes.insert(4, Route("/agents/status", self.agents_status, methods=["GET"]))
        app.routes.insert(5, Route("/agents/details", self.agent_details, methods=["GET"]))

        return app

    def health_check(self, request: Request) -> JSONResponse:
        """Return the health check for this agent."""
        return JSONResponse({"status": "ok", "agent": self.persona.name})

    async def chat_stream(self, request: Request) -> Response:
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
                    async for event in self.stream_chat_events_async(content):
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

    def memory_clear(self, request: Request) -> JSONResponse:
        """Return the memory clear for this agent."""
        return JSONResponse({"status": "ok", "agent": self.persona.name})

    def agents_status(self, request: Request) -> JSONResponse:
        """Return the agents status for this agent."""
        return JSONResponse({"status": "ok", "agent": self.persona.name})

    def agent_details(self, request: Request) -> JSONResponse:
        """Return the agent details for this agent."""
        return JSONResponse({"status": "ok", "agent": self.persona.name})

    def reset_session(self, *, session_id: str | None = None) -> None:
        """Reset the ADK session for a fresh run (useful for external transcripts)."""
        self.session_id = session_id or str(uuid4())
        self.session_ready = False

    def memory_status(self, request: Request) -> JSONResponse:
        """Return durable memory status for this agent."""
        transcript_path = None
        items = 0
        if self._transcript is not None:
            try:
                transcript_path = str(self._transcript.path)
                items = len(self._transcript.history)
            except Exception:
                transcript_path = None
                items = 0
        return JSONResponse(
            {
                "enabled": self._transcript is not None,
                "transcript_path": transcript_path,
                "items": items,
                "state_path": str(self._memory_state_path) if self._memory_state_path is not None else None,
            }
        )

    def clear_memory(self) -> None:
        """Clear durable transcript + state snapshot (best effort)."""
        if self._transcript is not None:
            try:
                self._transcript.clear()
            except Exception as e:
                logger.warning(f"Failed to clear transcript: {type(e).__name__}: {e}")
        if self._memory_state_path is not None and self._memory_state_path.exists():
            try:
                self._memory_state_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete memory state: {type(e).__name__}: {e}")

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
        # If durable transcript is enabled, we run each turn "stateless" and inject
        # bounded chat history as context. This preserves resume behavior across restarts
        # without relying on an on-disk ADK SessionService.
        injected_message = message
        user_text = ""
        try:
            parts = getattr(message, "parts", None) or []
            for part in parts:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt:
                    user_text += txt
            user_text = user_text.strip()
        except Exception:
            user_text = ""

        if self._transcript is not None and user_text:
            try:
                # Build a bounded dialogue preamble for context injection.
                history = self._transcript.as_dialog_text()
                preamble = ""
                if history:
                    preamble = (
                        "Context from prior conversation turns (most recent last):\n"
                        f"{history}\n\n"
                        "---\n\n"
                    )

                # Load a lightweight project-state snapshot (helps resume quickly).
                state_preamble = ""
                if self._memory_state_path is not None and self._memory_state_path.exists():
                    try:
                        raw = self._memory_state_path.read_text(encoding="utf-8")
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            updated_at = obj.get("updated_at", "")
                            topic = obj.get("topic", "")
                            files = obj.get("files", [])
                            if isinstance(files, list):
                                files = [f for f in files if isinstance(f, str) and f][:10]
                            else:
                                files = []
                            lines = ["Project state snapshot:"]
                            if isinstance(updated_at, str) and updated_at:
                                lines.append(f"- updated_at: {updated_at}")
                            if isinstance(topic, str) and topic:
                                lines.append(f"- topic: {topic}")
                            if files:
                                lines.append("- relevant_files:")
                                lines.extend([f"  - {f}" for f in files])
                            state_preamble = "\n".join(lines).strip() + "\n\n---\n\n"
                    except Exception:
                        state_preamble = ""

                # Persist the raw user message (after building preamble so we don't
                # duplicate it inside the injected context for this same turn).
                self._transcript.append_event(role="user", text=user_text)

                injected_message = types.Content(
                    role="user",
                    parts=[types.Part(text=state_preamble + preamble + user_text)],
                )

                # Avoid double-memory: don't let ADK accumulate history in-memory if
                # we're already injecting it from disk.
                self.reset_session()
            except Exception as e:
                logger.warning(f"Transcript injection failed; continuing without it: {type(e).__name__}: {e}")
                injected_message = message

        seen_text = ""
        assistant_text = ""
        try:
            async for event in self.run_events_async(injected_message):
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
                            assistant_text += txt
                            yield {"type": "text", "text": txt}
                            continue

                        # Case 1: cumulative replay (txt contains everything so far).
                        if txt.startswith(seen_text):
                            delta = txt[len(seen_text) :]
                            if delta:
                                seen_text = txt
                                assistant_text += delta
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
                        assistant_text += delta
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
        finally:
            # Persist assistant message at end of turn.
            if self._transcript is not None and assistant_text.strip():
                try:
                    self._transcript.append_event(role="assistant", text=assistant_text.strip())
                except Exception as e:
                    logger.warning(f"Failed to persist assistant transcript: {type(e).__name__}: {e}")

            # Update a tiny structured "project state" snapshot for quick resume.
            if self._memory_state_path is not None and user_text and assistant_text.strip():
                try:
                    text_blob = f"{user_text}\n{assistant_text}"
                    # Heuristic: capture paths that look like repo-relative files.
                    paths = sorted(
                        set(
                            m.group(1)
                            for m in re.finditer(
                                r"(?:^|\\s)([A-Za-z0-9_./-]+\\.(?:yml|yaml|py|md|txt|json|jsonl|tex|bib|pdf|png|jpg|jpeg))(?:$|[\\s,.;:])",
                                text_blob,
                            )
                        )
                    )
                    files = []
                    for p in paths:
                        p = p.strip()
                        if p.startswith(("artifacts/", "config/", "docs/", "ai/", "trainer/", "optimizer/", "layer/")):
                            files.append(p)
                    files = files[:25]

                    topic = (user_text.strip().splitlines()[0] if user_text.strip() else "")[:160]
                    state = {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "topic": topic,
                        "files": files,
                        "last_user": user_text[-2000:],
                        "last_assistant": assistant_text.strip()[-4000:],
                    }
                    self._memory_state_path.parent.mkdir(parents=True, exist_ok=True)
                    self._memory_state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to persist memory state: {type(e).__name__}: {e}")

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