"""Agent context for gathering information before responding."""
from __future__ import annotations

from dataclasses import dataclass, field
from agent.message import Message
from agent.knowledge import Knowledge


@dataclass
class AgentContext:
    """Context gathered by an agent before responding.

    Contains information from various sources that help the agent
    formulate a more informed response.
    """
    def __init__(self):
        self.history: list[Message] = []
        self.knowledge: list[Knowledge] = []

    def add_message(self, message: Message) -> None:
        """Add a message to the context."""
        self.history.append(message)

    def to_prompt(self) -> str:
        """Convert the context to a prompt."""
        out: list[str] = []

        out += "".join([
            "<history>",
            "\n\n".join([f"**{item.name}** ({item.role}):\n{item.content}" for item in self.history]),
            "</history>"
        ])

        out += "".join([
            "<knowledge>"
            "\n\n".join([f"**{item.name}** ({item.source}):\n{item.content}" for item in self.knowledge]),
            "</knowledge>"
        ])

        return "\n\n".join(out)