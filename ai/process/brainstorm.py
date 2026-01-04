"""Brainstorm process allows you to work with ChatGPT, Claude, and Gemini together

This puts you in a chat-like environment on the command line, where you can discuss
ongoing topics with your "team" of agents.
There are also various commands at your disposal to help you manage various workflows.
Often these commands will map to a Task, which in itself is a smaller process that
may or may not use short-lived AI agents.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
import random

from google.genai import types
from caramba.ai.process import Process
from caramba.ai.agent import Agent
from caramba.ai.process.transcript_store import TranscriptStore
from caramba.ai.process.brainstorm_turn import BrainstormTurnRunner
from caramba.ai.tasks.context_compaction import ContextCompactionTask
from caramba.ai.tasks.knowledge import KnowledgeExtractionTask
from caramba.ai.tasks.meeting_notes import MeetingNotesTask
from caramba.config.agents import MultiplexChatProcessConfig
from rich.console import Group
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule


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

    def __init__(self, *, agents: dict[str, Agent], process: MultiplexChatProcessConfig) -> None:
        super().__init__(agents, "brainstorm")
        self.process = process
        self.tasks = {
            "knowledge_extraction": KnowledgeExtractionTask(),
            "meeting_notes": MeetingNotesTask(),
            "context_compaction": ContextCompactionTask(),
        }
        self.name = process.user_name
        self.transcript = TranscriptStore(
            path=process.transcript_path,
            max_items=process.max_context_items,
            max_tokens=process.max_context_tokens,
            max_event_tokens=process.max_event_tokens,
            compact_after_bytes=process.compact_after_bytes,
        )
        self.ensure_history_loaded()

    def append(self, *, type: str, author: str, content: str) -> None:
        self.transcript.append_markdown_event(role=type, author=author, content=content)
        self.history = self.transcript.history

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

    def ensure_history_loaded(self) -> None:
        """Load persisted conversation history once per process run."""
        self.transcript.load()
        self.history = self.transcript.history

    async def run(self) -> None:
        """Run the brainstorm process"""
        self.logger.header(
            "Brainstorm",
            "Type '@chatgpt …', '@claude …', '@gemini …' or just chat to broadcast. "
            "Commands: /wipe (clear history), /collect (extract meeting notes), /recall (load most recent meeting notes)"
        )
        try:
            while True:
                try:
                    await self.handle_user_input(self.get_user_input().strip())
                except Exception as e:
                    # Keep the REPL alive even if a provider/tool/turn fails.
                    # The detailed error is usually already streamed into the UI; this is a final guardrail.
                    self.logger.error(str(e))
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

        # Keep diversity: sample a new temperature per agent-turn.
        # (This does not override the model/provider; it only changes sampling.)
        _ = random.random()

        runner = BrainstormTurnRunner(host=self, transcript=self.transcript)
        await self.compact_transcript_if_needed()
        streamed = await runner.run(agent=agent)
        # Some providers may emit only tool calls/results (no text parts). In that case,
        # we already recorded tool events to the transcript, so avoid appending an empty
        # assistant message (TranscriptStore enforces non-empty content).
        if isinstance(streamed, str) and streamed.strip():
            self.append(type="assistant", author=agent.persona.name, content=streamed)

    async def broadcast(self, user_input: str) -> None:
        """Broadcast the user input to all agents

        Always start from a randomly selected agent to prevent bias and inherent
        hierarchy of agents.
        """
        for agent in random.sample(
            list(self.agents.values()), len(self.agents)
        ):
            try:
                await self.next_agent(user_input, agent)
            except Exception as e:
                # Never let one agent/provider failure prevent other agents from responding.
                self.logger.error(f"{agent.persona.name}: {e}")
                continue

    async def compact_transcript_if_needed(self) -> None:
        """Compact transcript with an AI task when it exceeds the shared budget."""
        prompt = self.transcript.build_prompt()
        if self.transcript.token_count(prompt) <= self.process.max_context_tokens:
            return

        keep_last = 12
        if len(self.transcript.history) <= keep_last:
            return

        older = [m for m in self.transcript.history[:-keep_last] if m.role not in ("tool_call", "tool_result")]
        recent = self.transcript.history[-keep_last:]

        task: ContextCompactionTask = self.tasks["context_compaction"]
        task.history = older
        result = await task.run_async()
        memory = result.get("memory")
        if not isinstance(memory, str) or not memory.strip():
            raise RuntimeError("Context compaction failed: missing 'memory' output")

        self.transcript.history = [types.Content(role="system", parts=[types.Part(text=memory)])] + recent
        self.transcript.rewrite_file_from_history()
        self.history = self.transcript.history

    def clear_history(self) -> None:
        """Clear in-memory history and file-based log

        This helps remove alignment drift within a conversation.
        """
        self.transcript.clear()
        self.history = self.transcript.history


    async def handle_recall_command(self) -> None:
        """Handle the /recall command: load the most recent meeting notes file into conversation history."""
        meeting_notes_dir = Path("artifacts/ai/meeting_notes")

        if not meeting_notes_dir.exists():
            raise FileNotFoundError(
                "No meeting notes directory found at artifacts/ai/meeting_notes. "
                "Run /collect first to create meeting notes."
            )

        meeting_files = list(meeting_notes_dir.glob("meeting_notes_*.md"))

        if not meeting_files:
            raise FileNotFoundError(
                "No meeting notes files found under artifacts/ai/meeting_notes. "
                "Run /collect first to create meeting notes."
            )

        meeting_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        most_recent_file = meeting_files[0]

        notes_content = most_recent_file.read_text(encoding="utf-8")
        formatted_content = f"[Recalled meeting notes from {most_recent_file.name}]\n\n{notes_content}"
        self.append(type="user", author=self.name, content=formatted_content)
        self.logger.success(
            f"Recalled meeting notes from {most_recent_file.name} and added to conversation history."
        )