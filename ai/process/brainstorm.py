"""Brainstorm process allows you to work with ChatGPT, Claude, and Gemini together

You will basically function as a team.
"""

from __future__ import annotations
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from caramba.ai.process import Process
from caramba.ai.agent import Agent
from caramba.ai.tasks.knowledge import KnowledgeExtractionTask
from caramba.ai.tasks.meeting_notes import MeetingNotesTask
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except (ImportError, Exception):
    _TIKTOKEN_ENCODING = None
    _HAS_TIKTOKEN = False


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

    def __init__(self, agents: dict[str, Agent]):
        super().__init__(agents, "brainstorm")
        self.name = os.getenv("USER") or "user"
        self.max_context_items = 40
        # Token budget: leave room for prompt template (~500 tokens) and response (~5000 tokens)
        # Default to 150000 tokens to stay well under the 200000 limit
        self.max_context_tokens = 150000
        self.history_path = Path("artifacts") / "ai" / "brainstorm.jsonl"
        self._history_loaded = False

    def _parse_routed_input(self, user_input: str) -> tuple[str, str] | None:
        """Parse an explicit agent route like '@chatgpt hello' or '@claude: hello'."""
        s = (user_input or "").strip()
        if not s.startswith("@"):
            return None
        # First token: '@chatgpt' or '@chatgpt:' etc.
        token, _, rest = s.partition(" ")
        key = token[1:].rstrip(":").strip().lower()
        if not key:
            return None
        if key not in self.agents:
            return None
        return key, rest.strip()

    def _ensure_history_loaded(self) -> None:
        """Load persisted conversation history once per process run."""
        if self._history_loaded:
            return
        self._history_loaded = True

        if not self.history_path.exists():
            return
        try:
            loaded = 0
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        self.append_history(obj)
                        loaded += 1
                except Exception:
                    # Skip malformed lines (best-effort recovery).
                    continue
            if loaded:
                self.logger.info(f"Resumed {loaded} transcript events from {self.history_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load transcript history from {self.history_path}: {e}")

    def _persist_event(self, event: dict[str, object]) -> None:
        """Append a single event to the persisted JSONL transcript."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            # Don't kill the REPL for logging issues.
            self.logger.warning(f"Failed to persist transcript event: {e}")

    def _append(self, *, type: str, author: str, content: object) -> None:
        event: dict[str, object] = {
            "ts": time.time(),
            "type": type,
            "author": author,
            "content": content,
        }
        self.append_history(event)
        self._persist_event(event)

    def _truncate_content(self, content: str, max_tokens: int = 10000) -> str:
        """Truncate content to fit within token budget.

        If content exceeds max_tokens, truncates from the middle, keeping
        the beginning and end with a separator.
        """
        content_tokens = self._estimate_tokens(content)
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

    def _render_items(self, items: list[dict[str, object]]) -> str:
        lines: list[str] = []
        for msg in items:
            mtype = str(msg.get("type", ""))
            author = str(msg.get("author", ""))
            content = msg.get("content", "")
            if mtype == "user":
                # Truncate very long user messages (max 5000 tokens)
                if isinstance(content, str):
                    content = self._truncate_content(content, max_tokens=5000)
                lines.append(f"- **{author}**: {content}")
            elif mtype == "assistant":
                # Truncate very long assistant messages (max 10000 tokens)
                if isinstance(content, str):
                    content = self._truncate_content(content, max_tokens=10000)
                lines.append(f"- **{author}**: {content}")
            elif mtype == "tool_call":
                payload = content if isinstance(content, dict) else {"call": content}
                lines.append(
                    f"- **{author} tool call**\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                )
            elif mtype == "tool_result":
                # Keep tool results compact in the shared transcript.
                payload = content if isinstance(content, dict) else {"result": content}
                lines.append(
                    f"- **tool result**\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                )
            else:
                if isinstance(content, str):
                    content = self._truncate_content(str(content), max_tokens=5000)
                lines.append(f"- **{author}** ({mtype}): {content}")
        return "\n".join(lines).strip()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses tiktoken if available (cl100k_base encoding used by Claude),
        otherwise falls back to character-based estimation (~4 chars per token).
        """
        if _HAS_TIKTOKEN and _TIKTOKEN_ENCODING is not None:
            try:
                return len(_TIKTOKEN_ENCODING.encode(text))
            except Exception:
                pass
        # Fallback: rough estimate of ~4 characters per token
        return len(text) // 4

    def _estimate_item_tokens(self, item: dict[str, object]) -> int:
        """Estimate token count for a single history item without truncation."""
        mtype = str(item.get("type", ""))
        author = str(item.get("author", ""))
        content = item.get("content", "")

        if mtype == "user":
            content_str = str(content) if not isinstance(content, str) else content
            # Estimate: "- **{author}**: {content}"
            return self._estimate_tokens(f"- **{author}**: {content_str}")
        elif mtype == "assistant":
            content_str = str(content) if not isinstance(content, str) else content
            # Estimate: "- **{author}**: {content}"
            return self._estimate_tokens(f"- **{author}**: {content_str}")
        elif mtype == "tool_call":
            payload = content if isinstance(content, dict) else {"call": content}
            payload_str = json.dumps(payload, ensure_ascii=False)
            # Estimate: "- **{author} tool call**\n```json\n{payload}\n```"
            return self._estimate_tokens(f"- **{author} tool call**\n```json\n{payload_str}\n```")
        elif mtype == "tool_result":
            payload = content if isinstance(content, dict) else {"result": content}
            payload_str = json.dumps(payload, ensure_ascii=False)
            # Estimate: "- **tool result**\n```json\n{payload}\n```"
            return self._estimate_tokens(f"- **tool result**\n```json\n{payload_str}\n```")
        else:
            content_str = str(content)
            # Estimate: "- **{author}** ({mtype}): {content}"
            return self._estimate_tokens(f"- **{author}** ({mtype}): {content_str}")

    def _render_transcript_markdown(self) -> str:
        """Render transcript items as compact markdown, respecting token budget.

        Keeps the most recent items that fit within max_context_tokens.
        If items are too long, truncates from the beginning of the window.
        """
        # Start with the most recent items (up to max_context_items)
        candidate_items = self.history[-self.max_context_items :]

        if not candidate_items:
            return ""

        # Estimate tokens for the prompt template (without transcript)
        prompt_template = (
            "You are one of several AI agents in a shared conversation.\n"
            "You MUST take into account everything already said (including other agents and tool results).\n"
            "Respond in **Markdown**.\n\n"
            "## Conversation so far\n"
            "{transcript}\n\n"
            "## Now respond to the latest user message\n"
        )
        template_tokens = self._estimate_tokens(prompt_template)

        # Available tokens for the transcript
        available_tokens = self.max_context_tokens - template_tokens

        # Build transcript from the end, working backwards until we hit the token limit
        selected_items: list[dict[str, object]] = []
        current_tokens = 0

        for item in reversed(candidate_items):
            item_tokens = self._estimate_item_tokens(item)

            # Always include at least the most recent item (even if it's over budget)
            if not selected_items:
                selected_items.insert(0, item)
                current_tokens += item_tokens
            elif current_tokens + item_tokens <= available_tokens:
                selected_items.insert(0, item)
                current_tokens += item_tokens
            else:
                # Can't fit this item, stop here
                break

        return self._render_items(selected_items)

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to the system clipboard."""
        if not text:
            return
        try:
            if sys.platform == "darwin":
                process = subprocess.Popen(
                    "pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE
                )
                process.communicate(text.encode("utf-8"))
        except Exception as e:
            self.logger.warning(f"Failed to copy to clipboard: {e}")

    def _compact_json(self, obj: object, *, max_chars: int = 500) -> str:
        """Compact JSON-ish objects for display."""
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 3] + "..."

    def _summarize_tool_result(self, resp: object) -> dict[str, object]:
        """Summarize tool results to avoid dumping huge payloads."""
        if isinstance(resp, dict):
            keys = list(resp.keys())
            return {"type": "object", "keys": keys[:20], "preview": self._compact_json(resp, max_chars=700)}
        if isinstance(resp, list):
            return {"type": "array", "len": len(resp), "preview": self._compact_json(resp[:5], max_chars=700)}
        return {"type": type(resp).__name__, "preview": self._compact_json(resp, max_chars=700)}

    def _live_view(self, *, answer_md: str, tool_md: str) -> Group:
        """Combine answer + tool activity into one Live renderable."""
        tool_section = tool_md.strip() or "_(no tool activity)_"
        return Group(
            Markdown(answer_md or ""),
            Rule(title="[muted]Tool activity[/muted]", style="muted"),
            Markdown(tool_section),
        )

    def _compose_agent_prompt(self, user_input: str) -> str:
        """Compose a prompt that forces shared context across agents.

        Validates that the prompt fits within token limits and logs warnings if approaching limits.
        """
        transcript = self._render_transcript_markdown()
        prompt_template = (
            "You are one of several AI agents in a shared conversation.\n"
            "You MUST take into account everything already said (including other agents and tool results).\n"
            "Respond in **Markdown**.\n\n"
            "## Conversation so far\n"
            f"{transcript}\n\n"
            "## Now respond to the latest user message\n"
            f"{user_input}\n"
        )

        # Validate prompt size
        prompt_tokens = self._estimate_tokens(prompt_template)
        if prompt_tokens > self.max_context_tokens:
            self.logger.warning(
                f"Prompt exceeds token budget: {prompt_tokens} tokens > {self.max_context_tokens} limit. "
                "Consider using /wipe to clear history and extract knowledge."
            )
        elif prompt_tokens > self.max_context_tokens * 0.9:
            self.logger.info(
                f"Prompt approaching token limit: {prompt_tokens} tokens ({prompt_tokens / self.max_context_tokens * 100:.1f}% of budget)"
            )

        return prompt_template

    async def run(self) -> None:
        """Run the brainstorm process"""
        self._ensure_history_loaded()
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

        # Check for /wipe command
        if user_input.strip() == "/wipe":
            await self._handle_wipe_command()
            return

        # Check for /collect command
        if user_input.strip() == "/collect":
            await self._handle_collect_command()
            return

        # Check for /recall command
        if user_input.strip() == "/recall":
            await self._handle_recall_command()
            return

        self._ensure_history_loaded()

        # New user turn: make it explicit in the shared transcript.
        self._append(type="user", author=self.name, content=user_input)

        # Start capturing generated content (everything AFTER the user message)
        start_index = len(self.history)

        # Ensure each agent's private ADK session doesn't drift from the shared transcript.
        for a in self.agents.values():
            a.reset_session()

        route = self._parse_routed_input(user_input)
        if route is not None:
            key, message = route
            await self.next_agent(message, agent=self.agents[key])
        else:
            await self.broadcast(user_input)

        # Copy generated content to clipboard
        new_items = self.history[start_index:]
        if new_items:
            self._copy_to_clipboard(self._render_items(new_items))

    async def next_agent(self, user_input: str, agent: Agent | None = None) -> None:
        """Next agent"""
        if agent is None:
            self.logger.error(f"Agent not found: {user_input}")
            return

        # Add a little space between participants.
        self.logger.console.print()
        self.logger.subheader(agent.persona.name)

        prompt = self._compose_agent_prompt(user_input)
        # Per-response creativity: randomize temperature in [0, 1).
        # We keep the same temperature for the entire agent turn (including any follow-up call).
        turn_temperature = random.random()

        streamed = ""
        saw_tool_event = False
        tool_events: list[str] = []
        max_tool_events = 12
        # Stream markdown by continuously re-rendering the current buffer.
        # Some upstream libraries emit noisy stderr lines; capture and filter them.
        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            with Live(self._live_view(answer_md="", tool_md=""), console=self.logger.console, refresh_per_second=12) as live:
                # Make generator shutdown explicit to avoid:
                # - RuntimeError: aclose(): asynchronous generator is already running
                # - anyio cancel scope exit mismatches from generator finalizers
                try:
                    async with contextlib.aclosing(
                        agent.stream_chat_events_async(
                            prompt,
                            streaming_mode="sse",
                            temperature=turn_temperature,
                        )
                    ) as stream:
                        async for ev in stream:
                            et = ev.get("type")
                            if et == "text":
                                chunk = ev.get("text") or ""
                                if chunk:
                                    streamed += str(chunk)
                                    live.update(
                                        self._live_view(
                                            answer_md=streamed,
                                            tool_md="\n".join(tool_events[-max_tool_events:]),
                                        )
                                    )
                            elif et == "tool_call":
                                saw_tool_event = True
                                name = ev.get("name")
                                args = ev.get("args")
                                self._append(
                                    type="tool_call",
                                    author=agent.persona.name,
                                    content={"name": name, "args": args, "id": ev.get("id")},
                                )
                                tool_events.append(
                                    "\n".join(
                                        [
                                            f"- **call** `{name}`",
                                            "```json",
                                            self._compact_json(args, max_chars=900),
                                            "```",
                                        ]
                                    )
                                )
                                live.update(
                                    self._live_view(
                                        answer_md=streamed,
                                        tool_md="\n".join(tool_events[-max_tool_events:]),
                                    )
                                )
                            elif et == "tool_result":
                                saw_tool_event = True
                                name = ev.get("name")
                                resp = ev.get("response")
                                summary = self._summarize_tool_result(resp)
                                self._append(
                                    type="tool_result",
                                    author=agent.persona.name,
                                    # Keep the shared transcript compact (summary only).
                                    content={"name": name, "summary": summary, "id": ev.get("id")},
                                )
                                tool_events.append(
                                    "\n".join(
                                        [
                                            f"- **result** `{name}`",
                                            "```json",
                                            self._compact_json(summary, max_chars=900),
                                            "```",
                                        ]
                                    )
                                )
                                live.update(
                                    self._live_view(
                                        answer_md=streamed,
                                        tool_md="\n".join(tool_events[-max_tool_events:]),
                                    )
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

        response = streamed.strip()
        if not response:
            # Some models may return tool calls only; prompt once more for a final text answer.
            if saw_tool_event:
                followup = (
                    self._compose_agent_prompt(user_input)
                    + "\n\n"
                    + "You may NOT call tools now. Provide the final answer in Markdown, "
                      "explicitly using any tool results already in the transcript."
                )
                with Live(
                    self._live_view(answer_md="", tool_md="\n".join(tool_events[-max_tool_events:])),
                    console=self.logger.console,
                    refresh_per_second=12,
                ) as live:
                    try:
                        async with contextlib.aclosing(
                            agent.stream_text_async(
                                followup,
                                streaming_mode="sse",
                                temperature=turn_temperature,
                            )
                        ) as stream2:
                            async for chunk in stream2:
                                if chunk:
                                    response += str(chunk)
                                    live.update(
                                        self._live_view(
                                            answer_md=response,
                                            tool_md="\n".join(tool_events[-max_tool_events:]),
                                        )
                                    )
                    except asyncio.CancelledError:
                        return
            response = response.strip()
            if not response:
                self.logger.warning(f"{agent.persona.name} produced no output.")
                return

        # Persist assistant message for the next agent to read.
        self._append(type="assistant", author=agent.persona.name, content=response)

    async def broadcast(self, user_input: str) -> None:
        """Broadcast the user input to all agents"""
        for agent in random.sample(list(self.agents.values()), len(self.agents)):
            await self.next_agent(user_input, agent)

    def _clear_history(self) -> None:
        """Clear in-memory history and file-based log."""
        # Clear in-memory history
        self.history.clear()
        self._history_loaded = False

        # Clear file-based log
        if self.history_path.exists():
            try:
                self.history_path.unlink()
                self.logger.info(f"Cleared history file: {self.history_path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete history file: {e}")

    async def _handle_wipe_command(self) -> None:
        """Handle the /wipe command: extract knowledge, store it, then clear history."""
        # Ensure history is loaded before extraction
        self._ensure_history_loaded()

        # Run knowledge extraction task
        task = KnowledgeExtractionTask(
            conversation_history=self.history,
            graph_name="caramba_knowledge_base",
        )
        result = await task.run_async()

        # Clear history
        self.logger.info("Clearing conversation history...")
        self._clear_history()

        if result.get("knowledge_extracted"):
            self.logger.success("History wiped. Knowledge has been preserved in caramba_knowledge_base graph.")
        else:
            self.logger.success("History wiped.")

    async def _handle_collect_command(self) -> None:
        """Handle the /collect command: extract meeting notes and save as markdown."""
        # Ensure history is loaded before extraction
        self._ensure_history_loaded()

        if not self.history:
            self.logger.warning("No conversation history to collect. Start a conversation first.")
            return

        # Run meeting notes extraction task
        task = MeetingNotesTask(conversation_history=self.history)
        result = await task.run_async()

        if result.get("notes_extracted"):
            filepath = result.get("filepath", "unknown")
            num_ideas = result.get("num_ideas", 0)
            self.logger.success(
                f"Meeting notes collected: {num_ideas} ideas extracted and saved to {filepath}"
            )
        else:
            self.logger.warning("Failed to extract meeting notes. The conversation may be too short or unclear.")

    async def _handle_recall_command(self) -> None:
        """Handle the /recall command: load the most recent meeting notes file into conversation history."""
        meeting_notes_dir = Path("artifacts/ai/meeting_notes")
        
        if not meeting_notes_dir.exists():
            self.logger.warning("No meeting notes directory found. Run /collect first to create meeting notes.")
            return

        # Find all meeting notes markdown files
        meeting_files = list(meeting_notes_dir.glob("meeting_notes_*.md"))
        
        if not meeting_files:
            self.logger.warning("No meeting notes files found. Run /collect first to create meeting notes.")
            return

        # Sort by modification time (most recent first)
        meeting_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        most_recent_file = meeting_files[0]

        try:
            # Read the meeting notes content
            notes_content = most_recent_file.read_text(encoding="utf-8")
            
            # Ensure history is loaded before appending
            self._ensure_history_loaded()
            
            # Add the meeting notes to conversation history as a user message
            # Format it clearly so agents know this is recalled context
            formatted_content = f"[Recalled meeting notes from {most_recent_file.name}]\n\n{notes_content}"
            self._append(type="user", author=self.name, content=formatted_content)
            
            self.logger.success(f"Recalled meeting notes from {most_recent_file.name} and added to conversation history.")
        except Exception as e:
            self.logger.warning(f"Failed to recall meeting notes from {most_recent_file}: {e}")