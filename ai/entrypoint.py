"""Docker entrypoint for running agents.

Reads environment variables to determine agent type and configuration,
then starts the appropriate server.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    """Run the appropriate agent server based on environment variables."""
    persona_type = os.environ.get("PERSONA_TYPE", "")
    agent_role = os.environ.get("AGENT_ROLE", "worker")  # root, lead, worker
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    team_config = os.environ.get("TEAM_CONFIG", "default")

    if not persona_type:
        print("ERROR: PERSONA_TYPE environment variable must be set", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid import errors during Docker build
    from caramba.ai import run_root_server, run_lead_server, run_agent_server

    if agent_role == "root" or persona_type == "root":
        print(f"Starting ROOT agent on {host}:{port}")
        run_root_server(host=host, port=port, team_config=team_config)
    elif agent_role == "lead":
        print(f"Starting LEAD agent '{persona_type}' on {host}:{port}")
        run_lead_server(persona_type, host=host, port=port, team_config=team_config)
    else:
        print(f"Starting WORKER agent '{persona_type}' on {host}:{port}")
        run_agent_server(persona_type, host=host, port=port)


if __name__ == "__main__":
    main()
