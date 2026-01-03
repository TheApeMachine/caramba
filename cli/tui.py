"""Textual TUI for chatting with the Root agent via A2A.

This provides a chat-like interface with:
- Main chat viewport (streaming responses)
- Sidebar showing consulted experts
- Tool calls sidebar
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, Log, Static
from textual.binding import Binding


class ExpertStatus(Static):
    """Widget showing status of a single expert agent."""

    def __init__(self, expert_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.expert_name = expert_name
        self.status = "idle"

    def on_mount(self) -> None:
        self.update(f"[dim]{self.expert_name}[/dim]")

    def set_status(self, status: str) -> None:
        """Update expert status (idle, consulting, done, error)."""
        self.status = status
        if status == "consulting":
            self.update(f"[yellow]●[/yellow] {self.expert_name}")
        elif status == "done":
            self.update(f"[green]✓[/green] {self.expert_name}")
        elif status == "error":
            self.update(f"[red]✗[/red] {self.expert_name}")
        else:
            self.update(f"[dim]{self.expert_name}[/dim]")


class ExpertsSidebar(Vertical):
    """Sidebar showing which experts are being consulted."""

    def compose(self) -> ComposeResult:
        yield Label("Experts", classes="sidebar-title")
        yield Container(id="experts-list")

    def add_expert(self, name: str) -> None:
        """Add an expert to the list."""
        experts_list = self.query_one("#experts-list")
        expert_widget = ExpertStatus(name)
        experts_list.mount(expert_widget)

    def update_expert_status(self, name: str, status: str) -> None:
        """Update an expert's status."""
        experts_list = self.query_one("#experts-list")
        for widget in experts_list.children:
            if isinstance(widget, ExpertStatus) and widget.expert_name == name:
                widget.set_status(status)
                break


class ToolCallWidget(Static):
    """Widget showing a single tool call."""

    def __init__(self, tool_name: str, agent_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_name = tool_name
        self.agent_name = agent_name

    def on_mount(self) -> None:
        self.update(f"[dim]{self.agent_name}[/dim] → [cyan]{self.tool_name}[/cyan]")


class ToolsSidebar(Vertical):
    """Sidebar showing tool calls."""

    def compose(self) -> ComposeResult:
        yield Label("Tools", classes="sidebar-title")
        yield Container(id="tools-list")

    def add_tool_call(self, tool_name: str, agent_name: str) -> None:
        """Add a tool call to the list."""
        tools_list = self.query_one("#tools-list")
        tool_widget = ToolCallWidget(tool_name, agent_name)
        tools_list.mount(tool_widget)


class ChatViewport(Log):
    """Main chat viewport showing conversation."""

    def add_user_message(self, text: str) -> None:
        """Add a user message to the chat."""
        self.write(f"[bold cyan]You[/bold cyan]: {text}")

    def add_assistant_chunk(self, text: str) -> None:
        """Add an assistant response chunk (streaming)."""
        # For streaming, we'll append to the last line if it's incomplete.
        self.write(text, end="")


class RootChatApp(App):
    """Textual TUI for chatting with Root agent."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    .sidebar {
        width: 20;
        border-right: wide $primary;
        background: $surface;
    }

    .sidebar-title {
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
    }

    #chat-area {
        width: 1fr;
        layout: vertical;
    }

    #chat-viewport {
        height: 1fr;
        border: wide $primary;
    }

    #input-area {
        height: 3;
        border-top: wide $primary;
    }

    #input {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, root_agent_url: str = "http://localhost:9000", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.root_agent_url = root_agent_url
        self.chat_viewport: ChatViewport | None = None
        self.experts_sidebar: ExpertsSidebar | None = None
        self.tools_sidebar: ToolsSidebar | None = None
        self.current_response_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(classes="sidebar"):
                yield ExpertsSidebar(id="experts")
            with Vertical(id="chat-area"):
                yield ChatViewport(id="chat-viewport")
                with Horizontal(id="input-area"):
                    yield Input(placeholder="Type your message...", id="input")
            with Vertical(classes="sidebar"):
                yield ToolsSidebar(id="tools")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app."""
        self.chat_viewport = self.query_one("#chat-viewport", ChatViewport)
        self.experts_sidebar = self.query_one("#experts", ExpertsSidebar)
        self.tools_sidebar = self.query_one("#tools", ToolsSidebar)
        self.chat_viewport.write("[bold green]Connected to Root agent[/bold green]\n")
        self.chat_viewport.write("Type a message and press Enter to chat.\n\n")

    @on(Input.Submitted, "#input")
    async def handle_input(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message:
            return

        # Clear input
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""

        # Show user message
        if self.chat_viewport:
            self.chat_viewport.add_user_message(user_message)
            self.chat_viewport.write("\n")
            self.current_response_text = ""

        # Stream response from root agent
        await self.stream_root_response(user_message)

    @work(exclusive=True)
    async def stream_root_response(self, user_message: str) -> None:
        """Stream response from root agent via SSE."""
        if not self.chat_viewport:
            return

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Send message to root agent and stream response
                # Note: This is a simplified version; actual ADK A2A endpoints may differ
                async with client.stream(
                    "POST",
                    f"{self.root_agent_url}/chat",
                    json={"message": user_message},
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            try:
                                data = json.loads(data_str)
                                await self.handle_stream_event(data)
                            except json.JSONDecodeError:
                                # Handle plain text chunks
                                if self.chat_viewport:
                                    self.chat_viewport.write(data_str)
                                    self.current_response_text += data_str
        except Exception as e:
            if self.chat_viewport:
                self.chat_viewport.write(f"\n[red]Error: {e}[/red]\n")

    async def handle_stream_event(self, event: dict[str, Any]) -> None:
        """Handle a single stream event from the root agent."""
        event_type = event.get("type")
        if event_type == "text" and self.chat_viewport:
            text = event.get("text", "")
            self.chat_viewport.write(text)
            self.current_response_text += text
        elif event_type == "tool_call" and self.tools_sidebar:
            tool_name = event.get("name", "unknown")
            agent_name = event.get("agent", "root")
            self.tools_sidebar.add_tool_call(tool_name, agent_name)
        elif event_type == "expert_consulting" and self.experts_sidebar:
            expert_name = event.get("expert", "")
            self.experts_sidebar.update_expert_status(expert_name, "consulting")
        elif event_type == "expert_done" and self.experts_sidebar:
            expert_name = event.get("expert", "")
            self.experts_sidebar.update_expert_status(expert_name, "done")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main() -> None:
    """Main entrypoint for the TUI."""
    root_url = os.getenv("ROOT_AGENT_URL", "http://localhost:9000")
    app = RootChatApp(root_agent_url=root_url)
    app.run()


if __name__ == "__main__":
    main()
