"""Brainstorm process allows you to work with ChatGPT, Claude, and Gemini together

This puts you in a chat-like environment on the command line, where you can discuss
ongoing topics with your "team" of agents.
There are also various commands at your disposal to help you manage various workflows.
Often these commands will map to a Task, which in itself is a smaller process that
may or may not use short-lived AI agents.
"""
from __future__ import annotations
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import random
import time
from google.genai import types

from caramba.ai.process import Process
from caramba.ai.agent import Agent
from caramba.ai.tasks.knowledge import KnowledgeExtractionTask
from caramba.ai.tasks.meeting_notes import MeetingNotesTask
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule
import tiktoken

TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


class Brainstorm(Process):
    """Brainstorm process allows you to work with ChatGPT, Claude, and Gemini together

    This process follows the following workflow:

    1. Upon starting, it loads the persistent chat history from: artifacts/ai/brainstorm.jsonl
    2. It pre-seeds the chat history for this session with the loaded history
    3. It prompts the user for input
    4. If the user does not address a specific agent, the prompt is broadcast to all agents in a random order
    5. Each agent response is directly added to the chat history which is given to the next agent so they can respond to each other
    5. At any point an agent can make a tool call, which should be executed and the result added to the chat history
    6. After a tool call we always loop back to the agent that made the tool call
    7. All messages are logged to the persistent chat history
    8. The process repeats until the user exits
    """

    def __init__(self, agents: dict[str, Agent]) -> None:
        super().__init__(agents, "brainstorm")
        self.tasks = {
            "knowledge_extraction": KnowledgeExtractionTask(),
            "meeting_notes": MeetingNotesTask(),
        }
        self.name = os.getenv("USER") or "user"
        self.max_context_items = 40
        self.max_context_tokens = 128000
        self.history_path = Path("artifacts") / "ai" / "brainstorm.jsonl"
        self.history_loaded = False
        self.ensure_history_loaded()

    def ensure_history_loaded(self) -> None:
        """Load persisted conversation history once per process run."""
        if self.history_loaded:
            return

        if not self.history_path.exists():
            return
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.append_history(
                        types.Content(
                            role=line["role"],
                            parts=[types.Part(text=line["text"])]
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Failed to load transcript history from {self.history_path}: {e}")

    def persist_event(self, event: dict[str, object]) -> None:
        """Append a single event to the persisted JSONL transcript."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            # Don't kill the REPL for logging issues.
            self.logger.warning(f"Failed to persist transcript event: {e}")

    def append(self, *, type: str, author: str, content: str) -> None:
        event: types.Content = types.Content(
            role=type, parts=[types.Part(text=f"**{author}**: {content}")]
        )
        self.append_history(event)
        self.persist_event({"role": type, "text": f"**{author}**: {content}"})

    def truncate_content(self, content: str, max_tokens: int = 128000) -> str:
        """Truncate content to fit within token budget.

        If content exceeds max_tokens, truncates from the middle, keeping
        the beginning and end with a separator.
        """
        content_tokens = len(TIKTOKEN_ENCODING.encode(content))
        if content_tokens <= max_tokens:
            return content

        # Estimate characters per token for truncation
        chars_per_token = len(content) / max(content_tokens, 1)
        max_chars = int(max_tokens * chars_per_token * 1.1)  # 10% buffer

        if len(content) <= max_chars:
            return content

        # Truncate from middle, keeping start and end
        keep_start = max_chars // 2
        keep_end = max_chars - keep_start - len("\n\n[... truncated ...]\n\n")
        truncated = content[:keep_start] + "\n\n[... truncated ...]\n\n" + content[-keep_end:]
        return truncated

    def live_view(self, *, answer_md: str, tool_calls: list[str]) -> Group:
        """Combine answer + tool call indicators into one Live renderable.

        Only shows tool call names (not results) to keep output clean.
        """
        if tool_calls:
            tool_section = "\n".join(tool_calls)
            return Group(
                Rule(title="[muted]Tools[/muted]", style="muted"),
                Markdown(tool_section),
                Rule(style="muted"),
                Markdown(answer_md or ""),
            )
        return Group(Markdown(answer_md or ""))

    async def run(self) -> None:
        """Run the brainstorm process"""
        self.logger.header(
            "Brainstorm",
            "Type '@chatgpt …', '@claude …', '@gemini …' or just chat to broadcast. "
            "Commands: /wipe (clear history), /collect (extract meeting notes), /recall (load most recent meeting notes)"
        )
        try:
            while True:
                await self.handle_user_input(self.get_user_input().strip())
        except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("Exiting.")
            return

    def get_user_input(self) -> str:
        """Get the user input"""
        # Visually separate the next user turn from streamed agent output.
        self.logger.console.print()
        self.logger.console.print(Rule(title=f"[highlight]{self.name}[/highlight]", style="muted"))
        return Prompt.ask(f"[info]{self.name}[/info]")

    async def handle_user_input(self, user_input: str) -> None:
        """Handle the user input"""
        if not user_input:
            return

        match user_input.strip():
            case "/wipe":
                await self.tasks["knowledge_extraction"].run_async()
                self.clear_history()
                return
            case "/collect":
                await self.tasks["meeting_notes"].run_async()
                return
            case "/recall":
                await self.handle_recall_command()
                return
            case _:
                pass

        self.append(type="user", author=self.name, content=user_input)
        await self.broadcast(user_input)

    async def next_agent(self, user_input: str, agent: Agent | None = None) -> None:
        """Next agent"""
        if agent is None:
            self.logger.error(f"Agent not found: {user_input}")
            return

        # Clear visual separator between agents
        self.logger.console.print()
        self.logger.console.print(
            Rule(
                title=f"[highlight]◆ {agent.persona.name} ◆[/highlight]",
                style="highlight",
            )
        )

        # Keep diversity: sample a new temperature per agent-turn.
        # (This does not override the model/provider; it only changes sampling.)
        turn_temperature = random.random()

        streamed = ""
        saw_tool_event = False
        # Track tool call names only (for display)
        tool_call_names: list[str] = []
        max_tool_display = 8
        # Stream markdown by continuously re-rendering the current buffer.
        # Some upstream libraries emit noisy stderr lines; capture and filter them.
        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            with Live(
                self.live_view(answer_md="", tool_calls=[]),
                console=self.logger.console, refresh_per_second=12
            ) as live:
                # Make generator shutdown explicit to avoid:
                # - RuntimeError: aclose(): asynchronous generator is already running
                # - anyio cancel scope exit mismatches from generator finalizers
                try:
                    async with contextlib.aclosing(
                        agent.stream_chat_events_async(
                            message=types.Content(role="user", parts=[types.Part(text=user_input)]),
                        )
                    ) as stream:
                        async for event in stream:
                            et = event.get("type")
                            if et == "text":
                                chunk = event.get("text") or ""
                                if chunk:
                                    streamed += str(chunk)
                                saw_tool_event = True
                                name = event.get("name")
                                args = event.get("args")
                                self.append(
                                    type="tool_call",
                                    author=agent.persona.name,
                                    content=f"name: {name}, args: {args}, id: {event.get('id')}",
                                )
                                # Only show tool name in console (not args or results)
                                tool_call_names.append(f"• `{name}`")
                                live.update(
                                    self.live_view(
                                        answer_md=streamed,
                                        tool_calls=tool_call_names[-max_tool_display:],
                                    )
                                )
                            elif et == "tool_result":
                                name = event.get("name")
                                response = event.get("response")

                                self.append(
                                    type="tool_result",
                                    author=agent.persona.name,
                                    # Keep the shared transcript compact (summary only).
                                    content=f"name: {name}, response: {response}, id: {event.get('id')}",
                                )
                except asyncio.CancelledError:
                    # Treat cancellation (Ctrl-C / task cancellation) as a clean exit.
                    return

        # Re-emit any unexpected stderr, but drop known noisy lines.
        stderr_text = err_buf.getvalue()
        if stderr_text:
            filtered: list[str] = []
            for line in stderr_text.splitlines():
                if "non-text parts in the response" in line:
                    continue
                if "BaseAuthenticatedTool" in line and "[EXPERIMENTAL]" in line:
                    continue
                # MCP shutdown/auth noise shouldn't break the REPL.
                if "Session termination failed:" in line:
                    continue
                if "Attempted to exit cancel scope in a different task" in line:
                    continue
                if "aclose(): asynchronous generator is already running" in line:
                    continue
                filtered.append(line)
            if filtered:
                self.logger.warning("\n".join(filtered))

        # Persist assistant message for the next agent to read.
        self.append(type="assistant", author=agent.persona.name, content=streamed)

    async def broadcast(self, user_input: str) -> None:
        """Broadcast the user input to all agents

        Always start from a randomly selected agent to prevent bias and inherent
        hierarchy of agents.
        """
        for agent in random.sample(
            list(self.agents.values()), len(self.agents)
        ):
            await self.next_agent(user_input, agent)

    def clear_history(self) -> None:
        """Clear in-memory history and file-based log

        This helps remove alignment drift within a conversation.
        """
        self.history.clear()

        # Clear file-based log
        if self.history_path.exists():
            try:
                self.history_path.unlink()
                self.logger.info(f"Cleared history file: {self.history_path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete history file: {e}")


    async def handle_recall_command(self) -> None:
        """Handle the /recall command: load the most recent meeting notes file into conversation history."""
        meeting_notes_dir = Path("artifacts/ai/meeting_notes")

        if not meeting_notes_dir.exists():
            self.logger.warning("No meeting notes directory found. Run /collect first to create meeting notes.")
            return

        meeting_files = list(meeting_notes_dir.glob("meeting_notes_*.md"))

        if not meeting_files:
            self.logger.warning("No meeting notes files found. Run /collect first to create meeting notes.")
            return

        meeting_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        most_recent_file = meeting_files[0]

        try:
            notes_content = most_recent_file.read_text(encoding="utf-8")
            formatted_content = f"[Recalled meeting notes from {most_recent_file.name}]\n\n{notes_content}"
            self.append(type="user", author=self.name, content=formatted_content)
            self.logger.success(
                f"Recalled meeting notes from {most_recent_file.name} and added to conversation history."
            )
        except Exception as e:
            self.logger.warning(f"Failed to recall meeting notes from {most_recent_file}: {e}")