"""Command palette modal for the Caramba TUI.

This module provides a searchable command palette (Ctrl+P) for
quick access to all available commands.
"""
from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Static
from textual.message import Message
from textual.binding import Binding

from caramba.tui.commands import Command, command_registry


class CommandPaletteItem(Static):
    """A single item in the command palette."""

    DEFAULT_CSS = """
    CommandPaletteItem {
        height: 3;
        padding: 0 2;
        background: transparent;
        layout: horizontal;
    }

    CommandPaletteItem:hover {
        background: #2D2D3D;
    }

    CommandPaletteItem.selected {
        background: #7C3AED;
    }

    CommandPaletteItem .cmd-info {
        width: 1fr;
    }

    CommandPaletteItem .cmd-name {
        color: #A78BFA;
        text-style: bold;
    }

    CommandPaletteItem .cmd-desc {
        color: #6C7086;
    }

    CommandPaletteItem .cmd-shortcut {
        width: auto;
        color: #6C7086;
        text-align: right;
    }
    """

    class Selected(Message):
        """Message sent when this item is selected."""

        def __init__(self, command: Command) -> None:
            super().__init__()
            self.command = command

    def __init__(self, command: Command, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.command = command

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        """Update the display."""
        shortcut = f"[dim]{self.command.shortcut}[/]" if self.command.shortcut else ""
        self.update(
            f"[cyan bold]{self.command.display_name}[/]  "
            f"[dim]{self.command.description}[/]"
            f"  {shortcut}"
        )

    def on_click(self) -> None:
        """Handle click on item."""
        self.post_message(self.Selected(self.command))


class CommandPalette(ModalScreen):
    """Modal command palette for quick command access."""

    DEFAULT_CSS = """
    CommandPalette {
        align: center middle;
    }

    CommandPalette #palette-container {
        width: 70%;
        max-width: 80;
        height: auto;
        max-height: 28;
        background: #1E1E2E;
        border: round #7C3AED;
        padding: 1;
    }

    CommandPalette #palette-input {
        width: 100%;
        background: #2D2D3D;
        border: round #45475A;
        padding: 1;
        margin-bottom: 1;
    }

    CommandPalette #palette-input:focus {
        border: round #7C3AED;
    }

    CommandPalette #palette-results {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }

    CommandPalette .category-header {
        height: 2;
        padding: 0 2;
        color: #6C7086;
        text-style: bold italic;
        background: #2D2D3D;
    }

    CommandPalette .no-results {
        height: 3;
        padding: 1 2;
        color: #6C7086;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("up", "move_up", "Move up", show=False),
        Binding("down", "move_down", "Move down", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

    class CommandSelected(Message):
        """Message sent when a command is selected from the palette."""

        def __init__(self, command: Command) -> None:
            super().__init__()
            self.command = command

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._results: list[Command] = []
        self._selected_index = 0
        self._items: list[CommandPaletteItem] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-container"):
            yield Input(
                placeholder="Search commands...",
                id="palette-input",
            )
            with VerticalScroll(id="palette-results"):
                pass

    def on_mount(self) -> None:
        """Focus the input and show all commands."""
        self.query_one("#palette-input", Input).focus()
        self._show_results("")

    def _show_results(self, query: str) -> None:
        """Show search results for the query."""
        results_container = self.query_one("#palette-results", VerticalScroll)

        # Clear existing results
        for child in list(results_container.children):
            child.remove()
        self._items.clear()

        # Search commands
        self._results = command_registry.search(query)
        self._selected_index = 0

        if not self._results:
            results_container.mount(
                Static("[dim]No commands found[/]", classes="no-results")
            )
            return

        # Group by category
        categories: dict[str, list[Command]] = {}
        for cmd in self._results:
            if cmd.category not in categories:
                categories[cmd.category] = []
            categories[cmd.category].append(cmd)

        # Mount results grouped by category
        for category, commands in categories.items():
            results_container.mount(
                Static(f"[dim italic]{category.upper()}[/]", classes="category-header")
            )
            for cmd in commands:
                item = CommandPaletteItem(cmd)
                self._items.append(item)
                results_container.mount(item)

        # Highlight first item
        if self._items:
            self._items[0].add_class("selected")

    def _update_selection(self) -> None:
        """Update the visual selection."""
        for i, item in enumerate(self._items):
            if i == self._selected_index:
                item.add_class("selected")
            else:
                item.remove_class("selected")

    @on(Input.Changed, "#palette-input")
    def handle_input_change(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self._show_results(event.value)

    @on(Input.Submitted, "#palette-input")
    def handle_submit(self, event: Input.Submitted) -> None:
        """Handle enter key on input."""
        self.action_select()

    @on(CommandPaletteItem.Selected)
    def handle_item_selected(self, event: CommandPaletteItem.Selected) -> None:
        """Handle item selection via click."""
        self.post_message(self.CommandSelected(event.command))
        self.dismiss()

    def action_close(self) -> None:
        """Close the palette."""
        self.dismiss()

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._items and self._selected_index > 0:
            self._selected_index -= 1
            self._update_selection()

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._items and self._selected_index < len(self._items) - 1:
            self._selected_index += 1
            self._update_selection()

    def action_select(self) -> None:
        """Select the current item."""
        if self._items and 0 <= self._selected_index < len(self._items):
            command = self._items[self._selected_index].command
            self.post_message(self.CommandSelected(command))
            self.dismiss()


class HelpScreen(ModalScreen):
    """Help screen showing all available commands and shortcuts."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen #help-container {
        width: 80%;
        max-width: 90;
        height: 80%;
        max-height: 40;
        background: #1E1E2E;
        border: round #7C3AED;
        padding: 2;
    }

    HelpScreen .help-title {
        height: 3;
        text-style: bold;
        text-align: center;
        color: #A78BFA;
        margin-bottom: 1;
    }

    HelpScreen #help-content {
        height: 1fr;
        overflow-y: auto;
    }

    HelpScreen .help-section {
        margin-bottom: 2;
    }

    HelpScreen .help-section-title {
        text-style: bold;
        color: #F59E0B;
        margin-bottom: 1;
    }

    HelpScreen .help-item {
        height: auto;
        padding: 0 2;
    }

    HelpScreen .help-close-hint {
        height: 2;
        text-align: center;
        color: #6C7086;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static("📚 Help & Keyboard Shortcuts", classes="help-title")
            with VerticalScroll(id="help-content"):
                # Keyboard shortcuts section
                yield Static("[bold]⌨️  Keyboard Shortcuts[/]", classes="help-section-title")
                yield Static(
                    "[cyan]Ctrl+P[/]     Open command palette\n"
                    "[cyan]Ctrl+L[/]     Clear chat\n"
                    "[cyan]Ctrl+Q[/]     Quit application\n"
                    "[cyan]↑ / ↓[/]      Navigate history\n"
                    "[cyan]Escape[/]     Close popups/cancel\n"
                    "[cyan]Tab[/]        Accept autocomplete",
                    classes="help-item",
                )

                # Commands by category
                for category in command_registry.categories:
                    commands = command_registry.get_by_category(category)
                    if commands:
                        yield Static(
                            f"\n[bold]📁 {category.upper()}[/]",
                            classes="help-section-title",
                        )
                        lines = []
                        for cmd in commands:
                            shortcut = f" ({cmd.shortcut})" if cmd.shortcut else ""
                            lines.append(
                                f"[cyan]{cmd.display_name}[/]{shortcut}  [dim]{cmd.description}[/]"
                            )
                        yield Static("\n".join(lines), classes="help-item")

            yield Static("[dim]Press Escape or Q to close[/]", classes="help-close-hint")

    def action_close(self) -> None:
        """Close the help screen."""
        self.dismiss()
