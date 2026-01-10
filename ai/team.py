"""Team management for organizing agents into hierarchical groups.

Teams allow the root agent to delegate to team leads who then
coordinate with their team members.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import TeamConfig


class TeamLoader:
    """Loads and manages team configurations from YAML files.

    Teams define which agents work together and who leads each team.
    An agent can appear on multiple teams.
    """

    def __init__(self, teams_dir: str | Path | None = None) -> None:
        """Initialize the team loader.

        Args:
            teams_dir: Path to directory containing team YAML files.
                      Defaults to config/teams, checking multiple locations.
        """
        if teams_dir is None:
            # Check multiple possible locations (Docker mount, local dev, package)
            candidates = [
                Path("/app/config/teams"),  # Docker volume mount
                Path(__file__).parent.parent / "config" / "teams",  # Package-relative
                Path.cwd() / "config" / "teams",  # Working directory
            ]
            for candidate in candidates:
                if candidate.exists():
                    teams_dir = candidate
                    break
            else:
                # Default to package-relative if none exist
                teams_dir = Path(__file__).parent.parent / "config" / "teams"
        self.teams_dir = Path(teams_dir)
        self._cache: dict[str, dict[str, TeamConfig]] = {}

    def load(self, config_name: str = "default") -> dict[str, TeamConfig]:
        """Load teams from a configuration file.

        Args:
            config_name: The config file name (without extension).

        Returns:
            Dictionary mapping team name to TeamConfig.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        if config_name in self._cache:
            return self._cache[config_name]

        path = self.teams_dir / f"{config_name}.yml"
        if not path.exists():
            raise FileNotFoundError(f"Team config not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "teams" not in data:
            raise ValueError(f"Invalid team config: {path}")

        teams: dict[str, TeamConfig] = {}
        for team_name, team_data in data["teams"].items():
            teams[team_name] = TeamConfig.from_dict(team_name, team_data)

        self._cache[config_name] = teams
        return teams

    def get_leads(self, config_name: str = "default") -> set[str]:
        """Get all team leads from a configuration.

        Args:
            config_name: The config file name.

        Returns:
            Set of lead persona types.
        """
        teams = self.load(config_name)
        return {team.lead for team in teams.values()}

    def get_teams_for_agent(
        self, agent_type: str, config_name: str = "default"
    ) -> list[TeamConfig]:
        """Get all teams an agent belongs to.

        Args:
            agent_type: The persona type to look up.
            config_name: The config file name.

        Returns:
            List of teams the agent is on (as lead or member).
        """
        teams = self.load(config_name)
        result = []
        for team in teams.values():
            if team.lead == agent_type or agent_type in team.members:
                result.append(team)
        return result

    def get_members_for_lead(
        self, lead_type: str, config_name: str = "default"
    ) -> list[str]:
        """Get all team members for a lead.

        Args:
            lead_type: The lead persona type.
            config_name: The config file name.

        Returns:
            List of member persona types across all teams led by this agent.
        """
        teams = self.load(config_name)
        members: set[str] = set()
        for team in teams.values():
            if team.lead == lead_type:
                members.update(team.members)
        return list(members)


class TeamRegistry:
    """Registry for runtime team and agent relationships.

    Tracks which agents are available and their health status.
    """

    def __init__(self, loader: TeamLoader | None = None) -> None:
        """Initialize the registry.

        Args:
            loader: TeamLoader instance. Creates one if not provided.
        """
        self.loader = loader or TeamLoader()
        self._agent_urls: dict[str, str] = {}

    def register_agent(self, agent_type: str, url: str) -> None:
        """Register an agent's URL.

        Args:
            agent_type: The persona type.
            url: The agent's base URL.
        """
        self._agent_urls[agent_type] = url

    def get_agent_url(self, agent_type: str) -> str | None:
        """Get the URL for an agent.

        Args:
            agent_type: The persona type.

        Returns:
            The agent's URL or None if not registered.
        """
        return self._agent_urls.get(agent_type)

    def get_all_agent_urls(self) -> dict[str, str]:
        """Get all registered agent URLs.

        Returns:
            Dictionary mapping persona type to URL.
        """
        return self._agent_urls.copy()

    def build_hierarchy(
        self, config_name: str = "default"
    ) -> dict[str, dict[str, Any]]:
        """Build the team hierarchy for status reporting.

        Args:
            config_name: The config file name.

        Returns:
            Nested dictionary suitable for /agents/status endpoint.
        """
        teams = self.loader.load(config_name)
        result: dict[str, dict[str, Any]] = {}

        for team_name, team in teams.items():
            agents: dict[str, dict[str, Any]] = {}

            # Add lead
            lead_url = self._agent_urls.get(team.lead, "")
            agents[team.lead] = {
                "url": lead_url,
                "healthy": bool(lead_url),
                "is_lead": True,
            }

            # Add members
            for member in team.members:
                member_url = self._agent_urls.get(member, "")
                agents[member] = {
                    "url": member_url,
                    "healthy": bool(member_url),
                    "is_lead": False,
                }

            # Include all agent names as a flat list for TUI compatibility
            all_members = [team.lead] + list(team.members)

            result[team_name] = {
                "lead": team.lead,
                "agents": agents,
                "members": all_members,  # Flat list for TUI sidebar
            }

        return result
