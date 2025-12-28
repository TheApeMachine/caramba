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
    history: list[Message] = field(default_factory=list)
    knowledge: list[Knowledge] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        """Add a message to the context."""
        self.history.append(message)

    def to_prompt(self) -> str:
        """Convert the context to a prompt."""
        sections: list[str] = []

        history = "\n\n".join(
            [f"**{item.name}** ({item.role}):\n{item.content}" for item in self.history]
        )
        sections.append(f"<history>\n{history}\n</history>")

        knowledge = "\n\n".join(
            [f"**{item.name}** ({item.source}):\n{item.content}" for item in self.knowledge]
        )
        sections.append(f"<knowledge>\n{knowledge}\n</knowledge>")

        return "\n\n".join(sections)