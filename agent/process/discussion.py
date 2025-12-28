"""Discussion process for multi-agent research conversations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from agent.process import Process
from agent.context import AgentContext

if TYPE_CHECKING:
    from agent import Researcher


class Discussion(Process):
    """A discussion process where agents are given a topic to discuss.

    The research team leader is responsible for orchestrating the discussion,
    and hands off to the most relevant agent to generate the next response.
    This allows for a more natural conversation flow, versus just a round-robin of agents.
    """

    def __init__(
        self,
        agents: dict[str, Researcher],
        team_leader_key: str = "research_team_leader",
        prompts_dir: Path = Path("config/prompts"),
    ):
        """Initialize the discussion.

        Args:
            agents: Dictionary of agents for the discussion.
            team_leader_key: Key of the team leader agent in the agents dict.
            prompts_dir: Directory containing prompt templates.
        """
        super().__init__(agents)
        self.team_leader_key = team_leader_key
        if team_leader_key not in agents:
            raise ValueError(f"Team leader '{team_leader_key}' not found in agents")

        # Load prompts
        with open(prompts_dir / "discussion.yml", "r") as f:
            self.prompts = yaml.safe_load(f)["discussion"]

        # Setup handoffs: leader can handoff to anyone, everyone hands back to leader
        for key in self.agents:
            if key != self.team_leader_key:
                self.handoff(self.team_leader_key, key)
                self.handoff(key, self.team_leader_key)

    async def run(
        self,
        topic: str,
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Run the discussion on a topic.

        Args:
            topic: The topic to discuss.
            context: Optional initial context.

        Returns:
            Dictionary containing the transcript and conclusion.
        """
        leader = self.agents[self.team_leader_key]
        participants = ", ".join([a.agent.name for a in self.agents.values()])

        # Configure leader instructions from template
        leader.agent.instructions = self.prompts["instructions"].format(
            topic=topic,
            participants=participants,
            notes=""
        )

        # Start discussion with initial prompt
        initial_prompt = self.prompts["initial_prompt"].format(
            topic=topic,
            participants=participants
        )

        # The Runner.run will follow handoffs automatically
        result = await leader.run(initial_prompt, context)

        return {
            "transcript": result.to_input_list(),
            "conclusion": result.content
        }
