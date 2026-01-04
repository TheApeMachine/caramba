"""One-turn streaming runner for multiplex chat.

This isolates the streaming, tool-event handling, and stderr filtering concerns
from the Brainstorm process so the process remains small and composable.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any, Mapping, cast

from google.genai import types
from rich.console import RenderableType
from rich.live import Live
from rich.rule import Rule

from caramba.ai.agent import Agent
from caramba.ai.process.transcript_store import TranscriptStore
from caramba.console import Logger


class BrainstormTurnRunner:
    """Run a single streamed agent turn.

    Used by multiplex chat to:
    - reset provider session state
    - stream text deltas into a live markdown view
    - record tool call/result events into the transcript
    - filter known noisy stderr traces from upstream libraries
    """

    def __init__(self, *, host: Any, transcript: TranscriptStore) -> None:
        self.host = host
        self.transcript = transcript

    async def run(self, *, agent: Agent) -> str:
        """Run a single agent turn and return the assistant text."""
        self.print_agent_header(agent)
        err_buf = io.StringIO()
        tool_call_names: list[str] = []

        with contextlib.redirect_stderr(err_buf):
            streamed = await self.stream_agent_response(agent=agent, tool_call_names=tool_call_names)

        self.emit_filtered_stderr(err_buf.getvalue())
        return streamed

    def print_agent_header(self, agent: Agent) -> None:
        """Print the agent section separator."""
        self.host.logger.console.print()
        self.host.logger.console.print(
            Rule(
                title=f"[highlight]◆ {agent.persona.name} ◆[/highlight]",
                style="highlight",
            )
        )

    async def stream_agent_response(self, *, agent: Agent, tool_call_names: list[str]) -> str:
        """Stream the agent response, updating live view and transcript."""
        streamed = ""
        max_tool_display = 8

        agent.reset_session()
        prompt = self.transcript.build_prompt()

        first_view = cast(RenderableType, self.host.live_view(answer_md="", tool_calls=[]))
        with Live(first_view, console=self.host.logger.console, refresh_per_second=12) as live:
            try:
                async with contextlib.aclosing(
                    agent.stream_chat_events_async(
                        message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                    )
                ) as stream:
                    async for event in stream:
                        streamed = self.handle_event(
                            event=event,
                            agent=agent,
                            streamed=streamed,
                            tool_call_names=tool_call_names,
                        )
                        view = cast(
                            RenderableType,
                            self.host.live_view(
                                answer_md=streamed,
                                tool_calls=tool_call_names[-max_tool_display:],
                            ),
                        )
                        live.update(view)
            except asyncio.CancelledError:
                return ""

        return streamed

    def handle_event(
        self,
        *,
        event: Mapping[str, Any],
        agent: Agent,
        streamed: str,
        tool_call_names: list[str],
    ) -> str:
        """Handle one streamed event and return updated streamed text."""
        event_type = event.get("type")
        if event_type == "text":
            return self.handle_text_event(event=event, streamed=streamed)
        if event_type == "tool_call":
            return self.handle_tool_call_event(
                event=event,
                agent=agent,
                streamed=streamed,
                tool_call_names=tool_call_names,
            )
        if event_type == "tool_result":
            return self.handle_tool_result_event(event=event, agent=agent, streamed=streamed)

        return streamed

    def handle_text_event(self, *, event: Mapping[str, Any], streamed: str) -> str:
        """Handle a streamed text-delta event."""
        chunk = event.get("text") or ""
        return streamed + str(chunk)

    def handle_tool_call_event(
        self,
        *,
        event: Mapping[str, Any],
        agent: Agent,
        streamed: str,
        tool_call_names: list[str],
    ) -> str:
        """Handle a streamed tool-call event."""
        name = event.get("name")
        args = event.get("args")
        self.host.append(
            type="tool_call",
            author=agent.persona.name,
            content=f"name: {name}, args: {args}, id: {event.get('id')}",
        )
        tool_call_names.append(f"• `{name}`")
        return streamed

    def handle_tool_result_event(
        self,
        *,
        event: Mapping[str, Any],
        agent: Agent,
        streamed: str,
    ) -> str:
        """Handle a streamed tool-result event."""
        name = event.get("name")
        response = event.get("response")
        self.host.append(
            type="tool_result",
            author=agent.persona.name,
            content=f"name: {name}, response: {response}, id: {event.get('id')}",
        )
        return streamed

    def emit_filtered_stderr(self, stderr_text: str) -> None:
        """Emit unexpected stderr lines, dropping known noisy patterns."""
        if not stderr_text:
            return

        filtered = []
        for line in stderr_text.splitlines():
            if self.is_known_noisy_stderr(line):
                continue
            filtered.append(line)

        if filtered:
            self.host.logger.warning("\n".join(filtered))

    def is_known_noisy_stderr(self, line: str) -> bool:
        """Return True when a stderr line is known-noisy and safe to drop."""
        noisy_substrings = [
            "non-text parts in the response",
            "BaseAuthenticatedTool",
            "Session termination failed:",
            "Attempted to exit cancel scope in a different task",
            "aclose(): asynchronous generator is already running",
            "an error occurred during closing of asynchronous generator",
            "asyncgen:",
            "Exception Group Traceback",
            "mcp/client/sse.py",
            "mcp/client/streamable_http.py",
            "streamable_http.py",
            "streamablehttp_client",
            "httpx_sse/_api.py",
            "GeneratorExit",
            "generator didn't stop after athrow()",
        ]
        return any(s in line for s in noisy_substrings)

