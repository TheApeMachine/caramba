"""Manifest-driven agent process runner.

This is the execution bridge between:
- `config.manifest.Manifest` (YAML-driven configuration)
- `agent.process.*` workflows (e.g. Discussion)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from console.logger import get_logger
from config.manifest import Manifest
from config.mcp_registry import load_mcp_servers
from config.persona import load_persona

from agent import Researcher
from agent.process.discussion import Discussion


logger = get_logger()


def _manifest_artifacts_dir(manifest: Manifest, manifest_path: Path) -> Path:
    name = manifest.name or manifest_path.stem
    return Path("artifacts") / name / "agents"


def _find_process(manifest: Manifest, process_name: str) -> Any:
    if manifest.agents is None:
        raise ValueError("Manifest has no 'agents' section.")
    for proc in manifest.agents.processes:
        if proc.name == process_name:
            return proc
    raise ValueError(f"Agent process '{process_name}' not found in manifest.")


def _build_team(manifest: Manifest) -> tuple[dict[str, Researcher], dict[str, str]]:
    if manifest.agents is None:
        raise ValueError("Manifest has no 'agents' section.")
    mapping = dict(manifest.agents.team.root)
    team: dict[str, Researcher] = {}
    for key, persona_name in mapping.items():
        persona = load_persona(persona_name)
        team[key] = Researcher(persona)
    return team, mapping


def _preflight_personas_and_tools(team_mapping: dict[str, str]) -> None:
    """Validate persona configs exist and their MCP servers are registered."""
    registry = load_mcp_servers()
    for _, persona_name in team_mapping.items():
        persona = load_persona(persona_name)
        for server_name in persona.mcp_servers:
            if server_name not in registry:
                raise ValueError(
                    f"Persona '{persona_name}' references unknown MCP server '{server_name}'. "
                    "Define it in config/mcp_servers.yml."
                )


def dry_run_process(manifest: Manifest, *, process_name: str, manifest_path: Path) -> dict[str, Any]:
    """Validate process config and return a summary without executing."""
    proc = _find_process(manifest, process_name)
    team, team_mapping = _build_team(manifest)
    _preflight_personas_and_tools(team_mapping)

    # Process-specific checks
    if getattr(proc, "type", None) == "discussion":
        prompts_dir = Path(proc.prompts_dir)
        prompts_file = prompts_dir / "discussion.yml"
        if not prompts_file.exists():
            raise ValueError(f"Missing discussion prompts file at {prompts_file}")
        if proc.leader not in team:
            raise ValueError(
                f"Discussion leader key '{proc.leader}' not present in agents.team."
            )

    out_dir = _manifest_artifacts_dir(manifest, manifest_path)
    return {
        "ok": True,
        "process": {"name": proc.name, "type": proc.type},
        "team": team_mapping,
        "artifacts_dir": str(out_dir),
    }


async def _run_discussion(
    manifest: Manifest,
    *,
    process_name: str,
    manifest_path: Path,
) -> dict[str, Any]:
    proc = _find_process(manifest, process_name)
    team, team_mapping = _build_team(manifest)
    _preflight_personas_and_tools(team_mapping)

    prompts_dir = Path(proc.prompts_dir)
    discussion = Discussion(
        agents=team,
        team_leader_key=proc.leader,
        prompts_dir=prompts_dir,
    )
    result = await discussion.run(proc.topic, context=None)

    out_dir = _manifest_artifacts_dir(manifest, manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{proc.name}_{timestamp}.json"

    payload: dict[str, Any] = {
        "created_at": now.isoformat(),
        "manifest": {
            "name": manifest.name,
            "path": str(manifest_path),
            "notes": manifest.notes,
        },
        "process": {
            "name": proc.name,
            "type": proc.type,
            "leader": proc.leader,
            "topic": proc.topic,
            "prompts_dir": proc.prompts_dir,
            "max_rounds": getattr(proc, "max_rounds", None),
        },
        "team": team_mapping,
        "result": result,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Agent process complete • wrote {out_path}")
    return {"artifacts": {out_path.name: out_path}, "result": result}


def run_process(manifest: Manifest, *, process_name: str, manifest_path: Path) -> dict[str, Any]:
    """Run a named agent process from a manifest."""
    proc = _find_process(manifest, process_name)
    if proc.type == "discussion":
        return asyncio.run(_run_discussion(manifest, process_name=process_name, manifest_path=manifest_path))
    raise ValueError(f"Unsupported agent process type: {proc.type!r}")

