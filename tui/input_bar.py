"""Input bar with autocomplete for the Caramba TUI.

This module provides an enhanced input widget with:
- Slash command autocomplete
- Command history navigation
- Multi-line input support
"""
from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, Button, Static, OptionList
from textual.widgets.option_list import Option
from textual.message import Message
from textual.reactive import reactive
from textual.binding import Binding

from caramba.tui.commands import Command, command_registry


class AutocompleteItem(Static):
    """A single autocomplete suggestion item."""

    DEFAULT_CSS = """
    AutocompleteItem {
        height: 2;
        padding: 0 2;
        background: transparent;
    }

    AutocompleteItem:hover {
        background: #2D2D3D;
    }

    AutocompleteItem.selected {
        background: #7C3AED;
    }

    AutocompleteItem .command-name {
        color: #A78BFA;
        text-style: bold;
    }

    AutocompleteItem .command-desc {
        color: #6C7086;
    }
    """

    def __init__(self, command: Command, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.command = command

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold cyan]{self.command.display_name}[/bold cyan]  "
            f"[dim]{self.command.description}[/dim]"
        )


class AutocompleteDropdown(Vertical):
    """Dropdown menu for autocomplete suggestions."""

    DEFAULT_CSS = """
    AutocompleteDropdown {
        height: auto;
        max-height: 12;
        display: none;
        background: #1E1E2E;
        border: round #45475A;
        margin-bottom: 1;
        padding: 0;
        layer: autocomplete;
    }

    AutocompleteDropdown.visible {
        display: block;
    }

    AutocompleteDropdown OptionList {
        height: auto;
        max-height: 10;
        background: transparent;
        border: none;
        padding: 0;
    }

    AutocompleteDropdown OptionList > .option-list--option {
        padding: 0 2;
    }

    AutocompleteDropdown OptionList > .option-list--option-highlighted {
        background: #7C3AED;
    }
    """

    class CommandSelected(Message):
        """Message sent when a command is selected."""

        def __init__(self, command: Command, dropdown: "AutocompleteDropdown") -> None:
            super().__init__()
            self.command = command
            self._control = dropdown

        @property
        def control(self) -> "AutocompleteDropdown":
            """The AutocompleteDropdown that sent this message."""
            return self._control

    is_visible = reactive(False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands: list[Command] = []
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield OptionList(id="autocomplete-options")

    def watch_is_visible(self, is_visible: bool) -> None:
        """React to visibility changes."""
        self.set_class(is_visible, "visible")

    def show_suggestions(self, query: str) -> None:
        """Show suggestions matching the query."""
        self.commands = command_registry.search(query)
        self.selected_index = 0

        option_list = self.query_one("#autocomplete-options", OptionList)
        option_list.clear_options()

        if self.commands:
            for cmd in self.commands[:8]:  # Limit to 8 suggestions
                option_list.add_option(
                    Option(
                        f"[bold cyan]{cmd.display_name}[/bold cyan]  [dim]{cmd.description}[/dim]",
                        id=cmd.name,
                    )
                )
            self.is_visible = True
            option_list.highlighted = 0
        else:
            self.is_visible = False

    def hide(self) -> None:
        """Hide the dropdown."""
        self.is_visible = False

    def move_selection(self, delta: int) -> None:
        """Move the selection up or down."""
        if not self.commands:
            return

        option_list = self.query_one("#autocomplete-options", OptionList)
        current = option_list.highlighted or 0
        new_index = max(0, min(len(self.commands) - 1, current + delta))
        option_list.highlighted = new_index
        self.selected_index = new_index

    def get_selected_command(self) -> Command | None:
        """Get the currently selected command."""
        if not self.commands or self.selected_index >= len(self.commands):
            return None
        return self.commands[self.selected_index]

    @on(OptionList.OptionSelected, "#autocomplete-options")
    def handle_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option_id:
            command = command_registry.get(str(event.option_id))
            if command:
                self.post_message(self.CommandSelected(command, self))
                self.hide()


class InputBar(Vertical):
    """Enhanced input bar with autocomplete support."""

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        min-height: 5;
        max-height: 14;
        background: #1E1E2E;
        padding: 1 2;
        dock: bottom;
    }

    InputBar #input-wrapper {
        height: auto;
        min-height: 3;
        layout: horizontal;
        background: #2D2D3D;
        border: round #45475A;
        padding: 0 1;
    }

    InputBar #input-wrapper:focus-within {
        border: round #7C3AED;
    }

    InputBar #chat-input {
        background: transparent;
        border: none;
        height: auto;
        min-height: 1;
        width: 1fr;
        padding: 1;
    }

    InputBar #chat-input:focus {
        border: none;
    }

    InputBar #send-button {
        width: auto;
        min-width: 8;
        height: 3;
        background: #7C3AED;
        border: none;
        margin: 0 0 0 1;
    }

    InputBar #send-button:hover {
        background: #A78BFA;
    }

    InputBar .hint-text {
        height: 1;
        color: #6C7086;
        text-align: right;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("up", "history_prev", "Previous history", show=False),
        Binding("down", "history_next", "Next history", show=False),
        Binding("escape", "close_autocomplete", "Close autocomplete", show=False),
    ]

    class MessageSubmitted(Message):
        """Message sent when user submits a message."""

        def __init__(self, message: str, input_bar: "InputBar") -> None:
            super().__init__()
            self.message = message
            self._control = input_bar

        @property
        def control(self) -> "InputBar":
            """The InputBar that sent this message."""
            return self._control

    class CommandInvoked(Message):
        """Message sent when user invokes a slash command."""

        def __init__(self, command: Command, args: str, input_bar: "InputBar") -> None:
            super().__init__()
            self.command = command
            self.args = args
            self._control = input_bar

        @property
        def control(self) -> "InputBar":
            """The InputBar that sent this message."""
            return self._control

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[str] = []
        self.history_index = -1
        self.temp_input = ""

    def compose(self) -> ComposeResult:
        yield AutocompleteDropdown(id="autocomplete")
        with Horizontal(id="input-wrapper"):
            yield Input(
                placeholder="Type a message or / for commands...",
                id="chat-input",
            )
            yield Button("Send", id="send-button", variant="primary")
        yield Static("[dim]/ for commands • ↑↓ history • Ctrl+P palette[/dim]", classes="hint-text")

    @on(Input.Changed, "#chat-input")
    def handle_input_change(self, event: Input.Changed) -> None:
        """Handle input changes for autocomplete."""
        value = event.value
        autocomplete = self.query_one("#autocomplete", AutocompleteDropdown)

        if value.startswith("/"):
            # Extract command query (everything after /)
            query = value[1:].split(" ")[0]
            autocomplete.show_suggestions(query)
        else:
            autocomplete.hide()

    @on(Input.Submitted, "#chat-input")
    def handle_submit(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if not value:
            return

        # Add to history
        if not self.history or self.history[-1] != value:
            self.history.append(value)
        self.history_index = -1

        # Clear input
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""

        # Hide autocomplete
        autocomplete = self.query_one("#autocomplete", AutocompleteDropdown)
        autocomplete.hide()

        # Check if it's a command
        if value.startswith("/"):
            parts = value[1:].split(" ", 1)
            cmd_name = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            command = command_registry.get(cmd_name)
            if command:
                self.post_message(self.CommandInvoked(command, args, self))
                return

        # Regular message
        self.post_message(self.MessageSubmitted(value, self))

    @on(Button.Pressed, "#send-button")
    async def handle_send_button(self, event: Button.Pressed) -> None:
        """Handle send button click."""
        input_widget = self.query_one("#chat-input", Input)
        if input_widget.value.strip():
            # Trigger submit
            await input_widget.action_submit()

    @on(AutocompleteDropdown.CommandSelected, "#autocomplete")
    def handle_command_selected(self, event: AutocompleteDropdown.CommandSelected) -> None:
        """Handle command selection from autocomplete."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = f"/{event.command.name} "
        input_widget.cursor_position = len(input_widget.value)
        input_widget.focus()

    def on_key(self, event: Any) -> None:
        """Handle key events for navigation."""
        autocomplete = self.query_one("#autocomplete", AutocompleteDropdown)
        input_widget = self.query_one("#chat-input", Input)

        if autocomplete.is_visible:
            if event.key == "up":
                autocomplete.move_selection(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                autocomplete.move_selection(1)
                event.prevent_default()
                event.stop()
            elif event.key == "tab" or event.key == "enter":
                cmd = autocomplete.get_selected_command()
                if cmd:
                    input_widget.value = f"/{cmd.name} "
                    input_widget.cursor_position = len(input_widget.value)
                    autocomplete.hide()
                    event.prevent_default()
                    event.stop()
            elif event.key == "escape":
                autocomplete.hide()
                event.prevent_default()
                event.stop()
        else:
            # History navigation when autocomplete not visible
            if event.key == "up" and not input_widget.value:
                self.action_history_prev()
                event.prevent_default()
                event.stop()
            elif event.key == "down" and self.history_index >= 0:
                self.action_history_next()
                event.prevent_default()
                event.stop()

    def action_history_prev(self) -> None:
        """Navigate to previous history item."""
        if not self.history:
            return

        input_widget = self.query_one("#chat-input", Input)

        if self.history_index == -1:
            self.temp_input = input_widget.value
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1

        if 0 <= self.history_index < len(self.history):
            input_widget.value = self.history[self.history_index]
            input_widget.cursor_position = len(input_widget.value)

    def action_history_next(self) -> None:
        """Navigate to next history item."""
        input_widget = self.query_one("#chat-input", Input)

        if self.history_index == -1:
            return

        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            input_widget.value = self.history[self.history_index]
        else:
            self.history_index = -1
            input_widget.value = self.temp_input

        input_widget.cursor_position = len(input_widget.value)

    def action_close_autocomplete(self) -> None:
        """Close the autocomplete dropdown."""
        autocomplete = self.query_one("#autocomplete", AutocompleteDropdown)
        autocomplete.hide()

    def focus_input(self) -> None:
        """Focus the input widget."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.focus()
