"""Interactive multiplex chat process.

This is a lightweight REPL that lets you talk to multiple model-backed agents
using a single shared transcript context.

Usage (in the console):
  - Prefix a message with @chatgpt, @claude, or @gemini to pick the next responder.
  - If no @tag is provided, the message is broadcast to all routes sequentially.
  - Type /help for commands, /exit to quit.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import litellm
from rich.text import Text

from caramba.agent.process import Process
from caramba.console import logger
from caramba.agent import Researcher
from caramba.agent.tools.file import FileTool


class MultiplexChat(Process):
    """Interactive chat across multiple agents with shared context."""
    def __init__(self, agents: dict[str, Researcher]):
        super().__init__(agents)
        self.name = os.getenv("USER") or "user"
        self.file_tool = FileTool()
        self.log_path = Path(__file__).resolve().parents[2] / "artifacts" / "multiplex_chat.jsonl"
        self.debug_events = bool(os.getenv("CARAMBA_MULTIPLEX_DEBUG_EVENTS"))
        self.color_map = {
            "user": "green",
            "assistant": "cyan",
            "chatgpt": "blue",
            "claude": "purple",
            "gemini": "orange",
            "tool": "yellow",
        }

        self.messages: list[litellm.ResponseInputItemParam] = []
        if self.log_path.exists():
            try:
                self.messages = []
                for line in self.log_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    sanitized = self._sanitize_input_item(rec)
                    if sanitized is not None:
                        self.messages.append(sanitized)
            except Exception:
                logger.warning(f"Failed to load messages from log file, continuing: {self.log_path!s}")

    @staticmethod
    def _sanitize_input_item(item: object) -> litellm.ResponseInputItemParam | None:
        """
        Convert a log record into a provider-safe Responses API input item.

        OpenAI's Responses API rejects legacy/extra fields like `name` on input
        messages, so we keep a strict subset here.
        """
        if not isinstance(item, dict):
            return None

        # Tool outputs in Responses API input form
        if item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if isinstance(call_id, str) and call_id and isinstance(output, str):
                return {"type": "function_call_output", "call_id": call_id, "output": output}
            return None

        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant", "system", "developer") and isinstance(content, str):
            return {"role": role, "content": content}

        # Drop legacy chat.completions-only shapes (e.g. `role: tool`, `tool_calls`, etc.)
        return None

    def log(self, message: litellm.ResponseInputItemParam) -> None:
        self.messages.append(message)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    async def run(self):
        """Run the chat process."""
        while True:
            raw = input(f"\n{self.name}: ").strip()
            if not raw:
                continue
            if raw.lower() in ["/exit", "/quit", "/q", "/bye"]:
                logger.info("Exiting chat...")
                break
            if raw.lower() in ["/help", "/h"]:
                logger.info("Available commands: /exit, /quit, /q, /bye")
                continue

            route, message = self._parse_route(raw)
            # Store a clean transcript message (no @route prefix).
            self.log({"role": "user", "content": message})
            if route is None:
                await self.broadcast()
            else:
                await self.run_route(route)

    def _parse_route(self, raw: str) -> tuple[str | None, str]:
        """Return (route|None, message) where route=None means broadcast."""
        if raw.startswith("@"):
            parts = raw.split(" ", 1)
            if len(parts) == 2:
                tag, msg = parts[0], parts[1].strip()
                route = tag[1:].strip()
                if route in self.agents and msg:
                    return route, msg
        return None, raw

    async def broadcast(self) -> None:
        """Broadcast the *last* user message to all agents in random order."""
        for route in random.sample(list(self.agents.keys()), len(self.agents)):
            await self.run_route(route)

    def _tools(self) -> list[litellm.ToolParam]:
        return [
            self.file_tool.read_file_definition,
            self.file_tool.list_files_definition,
            self.file_tool.search_text_definition,
        ]

    async def run_route(self, route: str) -> None:
        """Run a single route, including any tool-call followups."""
        if route not in self.agents:
            logger.warning(f"Unknown route: {route}")
            return

        persona = self.agents[route].persona
        model = persona.model
        tools = self._tools()

        # Tool loop: keep calling the model until it stops asking for tools.
        while True:
            stream: litellm.ResponseStream = await litellm.aresponses(
                instructions=persona.instructions,
                model=model,
                input=self.messages,
                temperature=float(persona.temperature) if hasattr(persona, "temperature") else random.uniform(0, 1),
                tools=tools,
                stream=True,
            )
            tool_calls = await self.handle_response(stream, route=route)
            if not tool_calls:
                break
            self._execute_tool_calls(tool_calls)

    @staticmethod
    def _get(obj: object, key: str, default: object | None = None) -> object | None:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    async def handle_response(self, stream: litellm.ResponseStream, route: str) -> list[dict[str, str]]:
        assistant_chunks: list[str] = []
        tool_calls: list[dict[str, str]] = []
        printed_header = False

        async def handle_event(event: litellm.ResponseEvent) -> None:
            nonlocal printed_header
            if self.debug_events:
                logger.console.print(Text(str(event), style="dim"))

            etype = str(getattr(event, "type", "") or "")
            if etype == "response.reasoning_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    logger.console.print(Text(delta, style="dim"), end="", soft_wrap=True)
                return

            if etype == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if not delta:
                    return
                if not printed_header:
                    logger.console.print(Text(f"\n{route}: ", style=self.color_map[route]), end="", soft_wrap=True)
                    printed_header = True
                logger.console.print(Text(delta, style=self.color_map[route]), end="", soft_wrap=True)
                assistant_chunks.append(delta)
                return

            # Tool calling (Responses API streaming shape).
            # We primarily harvest tool calls from the finalized output item.
            if etype == "response.output_item.done":
                item = getattr(event, "item", None)
                itype = self._get(item, "type", None)
                if itype == "function_call":
                    name = self._get(item, "name", "") or ""
                    call_id = self._get(item, "call_id", "") or ""
                    arguments = self._get(item, "arguments", "") or ""
                    if isinstance(name, str) and isinstance(call_id, str) and isinstance(arguments, str) and name and call_id:
                        tool_calls.append({"name": name, "call_id": call_id, "arguments": arguments})
                return

        # LiteLLM may return either a sync iterator or an async iterator.
        if hasattr(stream, "__aiter__"):
            async for event in stream:  # type: ignore[misc]
                await handle_event(event)
        else:
            for event in stream:  # type: ignore[assignment]
                await handle_event(event)

        if assistant_chunks:
            self.log({
                "role": "assistant",
                "content": "".join([f"[{route}] ", *assistant_chunks])
            })

        return tool_calls

    def _execute_tool_calls(self, tool_calls: list[dict[str, str]]) -> None:
        """Execute tool calls and append function_call_output items to transcript."""
        for call in tool_calls:
            name = call.get("name", "")
            call_id = call.get("call_id", "")
            arg_s = call.get("arguments", "") or "{}"
            try:
                args = json.loads(arg_s) if isinstance(arg_s, str) and arg_s else {}
            except Exception:
                args = {}

            # Show tool invocation + output in the console (but keep transcript minimal/provider-safe).
            logger.console.print(Text(f"\n(tool) {name} {arg_s}", style=self.color_map["tool"]))
            result = self.file_tool.handle_tool_call(name, args if isinstance(args, dict) else {})
            try:
                rendered = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                rendered = str(result)
            if len(rendered) > 2000:
                rendered = rendered[:2000] + "\n... (truncated) ..."
            logger.console.print(Text(rendered, style=self.color_map["tool"]))

            item: litellm.ResponseInputItemParam = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
            self.log(item)