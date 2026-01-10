"""Team configuration for the agent system

Defines how personas are grouped into teams and which persona leads each team. This is
used to keep the delegation graph declarative and consistent across Root and team-lead
agents without hardcoding membership in multiple places.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TeamSpec(BaseModel):
    """Agent team specification

    Captures a team lead and the list of persona ids that can be orchestrated by that lead.
    This is the core unit used to build the delegation topology for team-based operation.
    """

    lead: str = Field(default="", description="Persona id of the team lead")
    members: list[str] = Field(default_factory=list, description="Persona ids of team members")


class TeamsConfig(BaseModel):
    """Collection of team definitions

    Holds a mapping from team id to a team specification. This is loaded from YAML and
    used at runtime to configure which team leads can delegate to which members.
    """

    teams: dict[str, TeamSpec] = Field(default_factory=dict)

    @staticmethod
    def load(path: Path | str = "config/teams/default.yml") -> "TeamsConfig":
        """Load team definitions from YAML

        Provides a single canonical loader for the default teams file. This keeps agent
        orchestration declarative and avoids duplicating team membership across persona YAMLs.
        """
        p = Path(path)
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return TeamsConfig.model_validate(payload)

    def team_ids_for_lead(self, lead: str) -> list[str]:
        """Find teams led by a given persona

        Returns the team ids where the `lead` matches the provided persona id. This is used
        to derive sub-agent lists for team-lead personas at runtime.
        """
        out: list[str] = []
        for team_id, spec in self.teams.items():
            if spec.lead == lead:
                out.append(team_id)
        return out

    def members_for_lead(self, lead: str) -> list[str]:
        """Compute the member set for a lead persona

        Returns the union of members across all teams led by the persona. This supports the
        “one persona can be in multiple teams” model while keeping lead membership derived.
        """
        members: list[str] = []
        for team_id in self.team_ids_for_lead(lead):
            members.extend(self.teams[team_id].members)
        return sorted({m for m in members if isinstance(m, str) and m.strip()})

