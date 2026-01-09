"""Enhanced viewport for the Caramba TUI.

This module provides the main chat viewport with:
- Styled message bubbles
- Markdown rendering
- Streaming response support
- Auto-scroll behavior
"""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.reactive import reactive

from caramba.tui.message import (
    MessageBubble,
    MessageType,
    StreamingMessage,
    ThinkingIndicator,
)


class WelcomeMessage(Static):
    """Welcome message shown when the chat is empty."""

    DEFAULT_CSS = """
    WelcomeMessage {
        width: 100%;
        height: auto;
        padding: 4 8;
        text-align: center;
    }

    WelcomeMessage .welcome-title {
        text-style: bold;
        color: #A78BFA;
        margin-bottom: 2;
    }

    WelcomeMessage .welcome-subtitle {
        color: #6C7086;
    }

    WelcomeMessage .welcome-tips {
        margin-top: 2;
        color: #6C7086;
    }
    """

    def welcome_text(self) -> str:
        return "[bold #FFFFFF]Welcome to Caramba[/]"

    def compose(self) -> ComposeResult:
        yield Static(
            f"""
[bold #A78BFA]╭─────────────────────────────────────────╮[/]
[bold #A78BFA]│            {self.welcome_text()}           │[/]
[bold #A78BFA]╰─────────────────────────────────────────╯[/]

[dim]The agentic machine learning research substrate.[/]

[dim]Tips:[/]
  [cyan]/[/] [dim]Start a slash command[/]
  [cyan]Ctrl+P[/] [dim]Open command palette[/]
  [cyan]↑ ↓[/] [dim]Navigate history[/]
  [cyan]Ctrl+L[/] [dim]Clear chat[/]
""",
            classes="welcome-content",
        )


class Viewport(VerticalScroll):
    """Main chat viewport for displaying messages."""

    DEFAULT_CSS = """
    Viewport {
        height: 1fr;
        background: #11111B;
        padding: 1 2;
    }

    Viewport:focus {
        border: none;
    }

    Viewport > VerticalScroll {
        scrollbar-gutter: stable;
    }
    """

    auto_scroll = reactive(True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._messages: list[MessageBubble] = []
        self._current_stream: StreamingMessage | None = None
        self._thinking_indicator: ThinkingIndicator | None = None
        self._is_empty = True

    def compose(self) -> ComposeResult:
        yield WelcomeMessage(id="welcome")

    def on_mount(self) -> None:
        """Initialize viewport."""
        pass

    def _hide_welcome(self) -> None:
        """Hide the welcome message."""
        if self._is_empty:
            try:
                welcome = self.query_one("#welcome", WelcomeMessage)
                welcome.remove()
            except Exception:
                pass
            self._is_empty = False

    def add_user_message(self, content: str) -> MessageBubble:
        """Add a user message to the viewport."""
        self._hide_welcome()

        message = MessageBubble(
            content=content,
            message_type=MessageType.USER,
            sender="You",
        )
        self._messages.append(message)
        self.mount(message)
        self._scroll_to_bottom()
        return message

    def add_assistant_message(self, content: str) -> MessageBubble:
        """Add an assistant message to the viewport."""
        self._hide_welcome()

        message = MessageBubble(
            content=content,
            message_type=MessageType.ASSISTANT,
            sender="Caramba",
        )
        self._messages.append(message)
        self.mount(message)
        self._scroll_to_bottom()
        return message

    def add_system_message(self, content: str) -> MessageBubble:
        """Add a system message to the viewport."""
        self._hide_welcome()

        message = MessageBubble(
            content=content,
            message_type=MessageType.SYSTEM,
        )
        self._messages.append(message)
        self.mount(message)
        self._scroll_to_bottom()
        return message

    def add_error_message(self, content: str) -> MessageBubble:
        """Add an error message to the viewport."""
        self._hide_welcome()

        message = MessageBubble(
            content=content,
            message_type=MessageType.ERROR,
        )
        self._messages.append(message)
        self.mount(message)
        self._scroll_to_bottom()
        return message

    def start_streaming(self) -> StreamingMessage:
        """Start a new streaming message."""
        self._hide_welcome()
        self._remove_thinking_indicator()

        stream = StreamingMessage()
        self._current_stream = stream
        self._messages.append(stream)
        self.mount(stream)
        self._scroll_to_bottom()
        return stream

    def stream_token(self, token: str) -> None:
        """Add a token to the current streaming message."""
        if self._current_stream:
            self._current_stream.stream_token(token)
            if self.auto_scroll:
                self._scroll_to_bottom()

    def end_streaming(self) -> MessageBubble | None:
        """End the current streaming message."""
        if self._current_stream:
            self._current_stream.finalize()
            msg = self._current_stream
            self._current_stream = None
            return msg
        return None

    def show_thinking(self) -> None:
        """Show the thinking indicator."""
        self._hide_welcome()

        if not self._thinking_indicator:
            indicator = ThinkingIndicator()
            self._thinking_indicator = indicator
            self.mount(indicator)
            self._scroll_to_bottom()

    def _remove_thinking_indicator(self) -> None:
        """Remove the thinking indicator if present."""
        if self._thinking_indicator:
            self._thinking_indicator.remove()
            self._thinking_indicator = None

    def hide_thinking(self) -> None:
        """Hide the thinking indicator."""
        self._remove_thinking_indicator()

    def clear_messages(self) -> None:
        """Clear all messages from the viewport."""
        for message in self._messages:
            message.remove()
        self._messages.clear()
        self._current_stream = None
        self._remove_thinking_indicator()

        # Show welcome again
        self._is_empty = True
        self.mount(WelcomeMessage(id="welcome"))

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the viewport."""
        if self.auto_scroll:
            self.scroll_end(animate=False)
