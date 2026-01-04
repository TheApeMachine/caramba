"""Chat message widgets for the Caramba TUI.

This module provides styled message bubbles for different types
of chat messages (user, assistant, system, error).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Markdown
from textual.reactive import reactive


class MessageType(Enum):
    """Types of chat messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    TOOL = "tool"
    THINKING = "thinking"


class MessageBubble(Vertical):
    """A styled chat message bubble."""

    DEFAULT_CSS = """
    MessageBubble {
        height: auto;
        margin: 1 0;
        padding: 0;
    }

    MessageBubble.user {
        margin-left: 12;
    }

    MessageBubble.user .message-box {
        background: #7C3AED;
        border: round #5B21B6;
    }

    MessageBubble.assistant {
        margin-right: 12;
    }

    MessageBubble.assistant .message-box {
        background: #1E1E2E;
        border: round #45475A;
    }

    MessageBubble.system {
        margin: 0 4;
    }

    MessageBubble.system .message-box {
        background: #2D2D3D;
        border: none;
        text-align: center;
    }

    MessageBubble.system .message-header {
        display: none;
    }

    MessageBubble.error .message-box {
        background: rgba(239, 68, 68, 0.15);
        border: round #EF4444;
    }

    MessageBubble.tool {
        margin: 0 8;
    }

    MessageBubble.tool .message-box {
        background: #2D2D3D;
        border: dashed #3B82F6;
    }

    MessageBubble.thinking {
        margin-right: 12;
    }

    MessageBubble.thinking .message-box {
        background: #1E1E2E;
        border: round #45475A;
    }

    MessageBubble .message-box {
        height: auto;
        padding: 1 2;
    }

    MessageBubble .message-header {
        height: 1;
        color: #6C7086;
        margin-bottom: 1;
    }

    MessageBubble .message-sender {
        text-style: bold;
    }

    MessageBubble .message-time {
        text-align: right;
        color: #6C7086;
    }

    MessageBubble .message-content {
        height: auto;
    }

    MessageBubble .message-content Markdown {
        margin: 0;
        padding: 0;
    }

    MessageBubble.thinking .thinking-dots {
        color: #F59E0B;
    }
    """

    content = reactive("")

    def __init__(
        self,
        content: str = "",
        message_type: MessageType = MessageType.ASSISTANT,
        sender: str | None = None,
        timestamp: datetime | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._content = content
        self.message_type = message_type
        self.sender = sender or self._default_sender()
        self.timestamp = timestamp or datetime.now()

        # Add type class
        self.add_class(message_type.value)

    def _default_sender(self) -> str:
        """Get default sender name based on message type."""
        return {
            MessageType.USER: "You",
            MessageType.ASSISTANT: "Caramba",
            MessageType.SYSTEM: "System",
            MessageType.ERROR: "Error",
            MessageType.TOOL: "Tool",
            MessageType.THINKING: "Caramba",
        }.get(self.message_type, "Unknown")

    def compose(self) -> ComposeResult:
        with Vertical(classes="message-box"):
            with Horizontal(classes="message-header"):
                yield Static(f"[bold]{self.sender}[/bold]", classes="message-sender")
                yield Static(
                    self.timestamp.strftime("%H:%M"),
                    classes="message-time",
                )
            if self.message_type == MessageType.THINKING:
                yield Static(
                    "[dim italic]Thinking[/dim italic] [yellow]•••[/yellow]",
                    classes="message-content thinking-dots",
                    id="thinking-content",
                )
            else:
                yield Markdown(self._content, classes="message-content", id="message-md")

    def update_content(self, new_content: str) -> None:
        """Update the message content (for streaming)."""
        self._content = new_content
        try:
            md_widget = self.query_one("#message-md", Markdown)
            md_widget.update(new_content)
        except Exception:
            pass

    def append_content(self, chunk: str) -> None:
        """Append to the message content (for streaming)."""
        self._content += chunk
        self.update_content(self._content)


class StreamingMessage(MessageBubble):
    """A message bubble that supports streaming content updates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("message_type", MessageType.ASSISTANT)
        super().__init__(*args, **kwargs)
        self._stream_buffer = ""

    def stream_token(self, token: str) -> None:
        """Add a token to the stream buffer and update display."""
        self._stream_buffer += token
        self.update_content(self._stream_buffer)

    def finalize(self) -> None:
        """Mark streaming as complete."""
        # Could add visual indicator that streaming is done
        pass


class ThinkingIndicator(Static):
    """Animated thinking indicator."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 3;
        padding: 1 2;
        margin: 1 0;
        margin-right: 12;
        background: #1E1E2E;
        border: round #45475A;
        color: #6C7086;
    }
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        """Start the animation."""
        self._timer = self.set_interval(0.3, self._animate)
        self._update_display()

    def _animate(self) -> None:
        """Animate the thinking dots."""
        self._frame = (self._frame + 1) % 4
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current animation frame."""
        dots = "•" * (self._frame + 1)
        padding = " " * (3 - self._frame)
        self.update(f"[dim italic]Caramba is thinking[/] [yellow]{dots}{padding}[/]")

    def on_unmount(self) -> None:
        """Stop the animation."""
        if self._timer:
            self._timer.stop()


class ToolCallMessage(MessageBubble):
    """Special message bubble for tool calls."""

    def __init__(
        self,
        tool_name: str,
        tool_input: str = "",
        agent_name: str = "root",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        content = f"**{agent_name}** → `{tool_name}`"
        if tool_input:
            content += f"\n```\n{tool_input}\n```"

        super().__init__(
            content=content,
            message_type=MessageType.TOOL,
            sender="Tool Call",
            *args,
            **kwargs,
        )
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.agent_name = agent_name
