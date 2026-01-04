"""Command definitions for the Caramba TUI.

This module defines all slash commands and their handlers,
providing a centralized place for command management.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class Command:
    """Represents a slash command in the TUI."""

    name: str
    description: str
    shortcut: str | None = None
    aliases: list[str] = field(default_factory=list)
    handler: Callable[..., Awaitable[None]] | None = None
    category: str = "general"

    def matches(self, query: str) -> bool:
        """Check if this command matches the given query."""
        query = query.lower().lstrip("/")
        if self.name.lower().startswith(query):
            return True
        for alias in self.aliases:
            if alias.lower().startswith(query):
                return True
        return False

    @property
    def display_name(self) -> str:
        """Get display name with slash prefix."""
        return f"/{self.name}"


# Default commands for the TUI
DEFAULT_COMMANDS: list[Command] = [
    Command(
        name="help",
        description="Show available commands and keyboard shortcuts",
        shortcut="?",
        aliases=["h", "?"],
        category="general",
    ),
    Command(
        name="clear",
        description="Clear the chat history",
        shortcut="Ctrl+L",
        aliases=["cls", "reset"],
        category="chat",
    ),
    Command(
        name="export",
        description="Export chat to file",
        aliases=["save"],
        category="chat",
    ),
    Command(
        name="model",
        description="Switch AI model",
        aliases=["m"],
        category="settings",
    ),
    Command(
        name="temperature",
        description="Set response temperature (0.0-2.0)",
        aliases=["temp", "t"],
        category="settings",
    ),
    Command(
        name="system",
        description="Set system prompt",
        aliases=["sys"],
        category="settings",
    ),
    Command(
        name="experts",
        description="List available expert agents",
        aliases=["agents", "e"],
        category="agents",
    ),
    Command(
        name="tools",
        description="List available tools",
        aliases=["t"],
        category="agents",
    ),
    Command(
        name="connect",
        description="Connect to a different agent endpoint",
        aliases=["conn"],
        category="connection",
    ),
    Command(
        name="status",
        description="Show connection status",
        aliases=["stat", "s"],
        category="connection",
    ),
    Command(
        name="theme",
        description="Switch color theme",
        aliases=["color"],
        category="appearance",
    ),
    Command(
        name="compact",
        description="Toggle compact mode",
        aliases=["mini"],
        category="appearance",
    ),
    Command(
        name="history",
        description="Show command history",
        aliases=["hist"],
        category="general",
    ),
    Command(
        name="debug",
        description="Toggle debug mode",
        aliases=["dbg"],
        category="debug",
    ),
    Command(
        name="quit",
        description="Exit the application",
        shortcut="Ctrl+Q",
        aliases=["q", "exit"],
        category="general",
    ),
]


class CommandRegistry:
    """Registry for managing TUI commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._categories: dict[str, list[Command]] = {}

        # Register default commands
        for cmd in DEFAULT_COMMANDS:
            self.register(cmd)

    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command

        if command.category not in self._categories:
            self._categories[command.category] = []
        self._categories[command.category].append(command)

    def get(self, name: str) -> Command | None:
        """Get a command by name or alias."""
        name = name.lower().lstrip("/")

        # Direct match
        if name in self._commands:
            return self._commands[name]

        # Alias match
        for cmd in self._commands.values():
            if name in cmd.aliases:
                return cmd

        return None

    def search(self, query: str) -> list[Command]:
        """Search for commands matching a query."""
        if not query:
            return list(self._commands.values())

        results = []
        for cmd in self._commands.values():
            if cmd.matches(query):
                results.append(cmd)

        # Sort by relevance (exact match first, then alphabetically)
        query_lower = query.lower().lstrip("/")
        results.sort(key=lambda c: (
            0 if c.name.lower() == query_lower else 1,
            c.name.lower()
        ))

        return results

    def get_by_category(self, category: str) -> list[Command]:
        """Get all commands in a category."""
        return self._categories.get(category, [])

    @property
    def categories(self) -> list[str]:
        """Get all command categories."""
        return list(self._categories.keys())

    @property
    def all_commands(self) -> list[Command]:
        """Get all registered commands."""
        return list(self._commands.values())


# Global command registry instance
command_registry = CommandRegistry()
