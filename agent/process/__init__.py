"""Process base class for agent workflows.

A Process is a reusable workflow pattern that an agent can execute.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING

from agents import HandoffInputData, handoff
from agent.context import AgentContext

if TYPE_CHECKING:
    from agent import Researcher


class Process(ABC):
    """Base class for agent processes.

    A process defines a workflow pattern that can be executed by an agent.
    Subclasses implement specific workflows like discussions, reviews, etc.
    """

    def __init__(self, agents: dict[str, Researcher]):
        """Initialize the process.

        Args:
            agents: Dict of Researcher instances for the workflow.
        """
        self.agents: dict[str, Researcher] = agents

    def next_agent(self, agent: str) -> Researcher:
        """Get the next agent to respond and handoff to

        Args:
            agent: The current agent.
            context: The context of the conversation.

        Returns:
            The next agent to respond and handoff to.
        """
        return self.agents[agent]

    def handoff(
        self,
        from_agent: str,
        to_agent: str,
        input_filter: Optional[Callable[[HandoffInputData], HandoffInputData]] = None,
    ) -> None:
        """Setup a handoff from one agent to another.

        Args:
            from_agent: Key of the agent to handoff from.
            to_agent: Key of the agent to handoff to.
            input_filter: Optional filter function for handoff messages.
        """
        source = self.agents[from_agent].agent
        target = self.agents[to_agent].agent

        if source.handoffs is None:
            source.handoffs = []

        source.handoffs.append(handoff(target, input_filter=input_filter))
