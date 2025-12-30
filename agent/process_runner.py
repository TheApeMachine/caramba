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
from typing import Any, cast

from console.logger import get_logger
from config.agents import (
    CodeGraphSyncProcessConfig,
    DiscussionProcessConfig,
    PaperReviewProcessConfig,
    PaperWriteProcessConfig,
    PlatformImproveProcessConfig,
    ResearchLoopProcessConfig,
)
from config.manifest import Manifest
from config.target import ProcessTargetConfig
from config.mcp_registry import load_mcp_servers
from config.persona import load_persona

from agent import Researcher
from agent.process.discussion import Discussion


logger = get_logger()


def _manifest_artifacts_dir(manifest: Manifest, manifest_path: Path) -> Path:
    name = manifest.name or manifest_path.stem
    return Path("artifacts") / name / "agents"


def _persist_process_artifact(
    *,
    manifest: Manifest,
    manifest_path: Path | None,
    proc_name: str,
    result: dict[str, Any],
) -> Path:
    mp = manifest_path or Path(f"{manifest.name or 'manifest'}.yml")
    out_dir = _manifest_artifacts_dir(manifest, mp)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{str(proc_name)}_{timestamp}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Agent process complete • wrote {out_path}")
    return out_path


def _build_team(team_mapping: dict[str, str]) -> tuple[dict[str, Researcher], dict[str, str]]:
    team: dict[str, Researcher] = {}
    for key, persona_name in team_mapping.items():
        persona = load_persona(persona_name)
        team[key] = Researcher(persona)
    return team, team_mapping


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


def dry_run_process_target(
    manifest: Manifest, *, target: ProcessTargetConfig, manifest_path: Path | None
) -> dict[str, Any]:
    """Validate process config and return a summary without executing."""
    proc = target.process
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    # Process-specific checks
    if isinstance(proc, DiscussionProcessConfig):
        prompts_dir = Path(proc.prompts_dir)
        prompts_file = prompts_dir / "discussion.yml"
        if not prompts_file.exists():
            raise ValueError(f"Missing discussion prompts file at {prompts_file}")
        if proc.leader not in team:
            raise ValueError(
                f"Discussion leader key '{proc.leader}' not present in agents.team."
            )
    elif isinstance(proc, PaperWriteProcessConfig):
        if proc.writer not in team:
            raise ValueError(
                f"paper_write writer key '{proc.writer}' not present in agents.team."
            )
    elif isinstance(proc, PaperReviewProcessConfig):
        if proc.reviewer not in team:
            raise ValueError(
                f"paper_review reviewer key '{proc.reviewer}' not present in agents.team."
            )
    elif isinstance(proc, ResearchLoopProcessConfig):
        missing: list[str] = []
        for k in (proc.leader, proc.writer, proc.reviewer):
            if k not in team:
                missing.append(str(k))
        if missing:
            raise ValueError(
                "research_loop missing agent keys in agents.team: " + ", ".join(missing)
            )
    elif isinstance(proc, CodeGraphSyncProcessConfig):
        if proc.agent not in team:
            raise ValueError(
                f"code_graph_sync agent key '{proc.agent}' not present in agents.team."
            )

    mp = manifest_path or Path(f"{manifest.name or 'manifest'}.yml")
    out_dir = _manifest_artifacts_dir(manifest, mp)
    return {
        "ok": True,
        "process": {"name": proc.name, "type": proc.type},
        "team": team_mapping,
        "artifacts_dir": str(out_dir),
    }


async def _run_discussion(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    proc = cast(DiscussionProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    prompts_dir = Path(proc.prompts_dir)
    discussion = Discussion(
        agents=team,
        team_leader_key=proc.leader,
        prompts_dir=prompts_dir,
    )
    result = await discussion.run(proc.topic, context=None)

    now = datetime.now(timezone.utc)
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
    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=payload
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


async def _run_paper_write(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    # Local import so `caramba` can import without the process module present during refactors.
    from agent.process.paper_write import PaperWrite  # type: ignore[import-not-found]

    proc = cast(PaperWriteProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    p = PaperWrite(agents=team, writer_key=proc.writer, output_dir=proc.output_dir)
    result = await p.run(manifest=manifest, manifest_path=manifest_path, goal=str(proc.goal))

    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=result
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


async def _run_paper_review(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    from agent.process.paper_review import PaperReview  # type: ignore[import-not-found]

    proc = cast(PaperReviewProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    p = PaperReview(
        agents=team,
        reviewer_key=proc.reviewer,
        strictness=str(proc.strictness),
        max_proposed_experiments=int(proc.max_proposed_experiments),
        output_dir=str(getattr(proc, "output_dir", "paper")),
    )
    result = await p.run(manifest=manifest, manifest_path=manifest_path, goal=str(proc.goal))

    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=result
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


async def _run_research_loop(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    from agent.process.research_loop import ResearchLoopProcess  # type: ignore[import-not-found]

    proc = cast(ResearchLoopProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    loop = ResearchLoopProcess(
        agents=team,
        leader_key=proc.leader,
        writer_key=proc.writer,
        reviewer_key=proc.reviewer,
        max_iterations=int(proc.max_iterations),
        auto_run_experiments=bool(proc.auto_run_experiments),
        output_dir=str(proc.output_dir),
    )
    result = await loop.run(manifest=manifest, manifest_path=manifest_path)

    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=result
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


async def _run_code_graph_sync(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    from agent.process.code_graph_sync import CodeGraphSync  # type: ignore[import-not-found]

    proc = cast(CodeGraphSyncProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    sync = CodeGraphSync(
        agents=team,
        agent_key=proc.agent,
        index_namespace=str(proc.index_namespace),
    )
    result = await sync.run(manifest=manifest, manifest_path=manifest_path)

    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=result
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


async def _run_platform_improve(
    manifest: Manifest,
    *,
    target: ProcessTargetConfig,
    manifest_path: Path | None,
) -> dict[str, Any]:
    # Local import so the runner stays resilient to optional process modules.
    from agent.process.platform_improve import PlatformImprove  # type: ignore[import-not-found]

    proc = cast(PlatformImproveProcessConfig, target.process)
    team_mapping = dict(target.team.root)
    team, _ = _build_team(team_mapping)
    _preflight_personas_and_tools(team_mapping)

    p = PlatformImprove(
        agents=team,
        ingest_agent=str(proc.ingest_agent),
        index_namespace=str(proc.index_namespace),
        ingest_repo=bool(proc.ingest_repo),
        ingest_models=bool(proc.ingest_models),
        max_files=int(proc.max_files),
        max_chars_per_file=int(proc.max_chars_per_file),
        leader_key=str(proc.leader),
        ideator_keys=[str(x) for x in list(proc.ideators)],
        developer_key=str(proc.developer),
        reviewer_key=str(proc.reviewer),
        repo_root=str(proc.repo_root),
        base_branch=str(proc.base_branch),
        branch_prefix=str(proc.branch_prefix),
        tests=[str(x) for x in list(proc.tests)],
        max_review_rounds=int(proc.max_review_rounds),
        open_pr=bool(proc.open_pr),
        pr_title_prefix=str(proc.pr_title_prefix),
        topic=str(proc.topic),
    )
    result = await p.run(manifest=manifest, manifest_path=manifest_path)

    out_path = _persist_process_artifact(
        manifest=manifest, manifest_path=manifest_path, proc_name=proc.name, result=result
    )
    return {"artifacts": {out_path.name: out_path}, "result": result}


def run_process_target(
    *, manifest: Manifest, target: ProcessTargetConfig, manifest_path: Path | None
) -> dict[str, Any]:
    """Run a process target."""
    proc = target.process
    if isinstance(proc, DiscussionProcessConfig):
        return asyncio.run(_run_discussion(manifest, target=target, manifest_path=manifest_path))
    if isinstance(proc, PaperWriteProcessConfig):
        return asyncio.run(_run_paper_write(manifest, target=target, manifest_path=manifest_path))
    if isinstance(proc, PaperReviewProcessConfig):
        return asyncio.run(_run_paper_review(manifest, target=target, manifest_path=manifest_path))
    if isinstance(proc, ResearchLoopProcessConfig):
        return asyncio.run(_run_research_loop(manifest, target=target, manifest_path=manifest_path))
    if isinstance(proc, CodeGraphSyncProcessConfig):
        return asyncio.run(_run_code_graph_sync(manifest, target=target, manifest_path=manifest_path))
    if isinstance(proc, PlatformImproveProcessConfig):
        return asyncio.run(_run_platform_improve(manifest, target=target, manifest_path=manifest_path))
    raise ValueError(f"Unsupported agent process type: {proc.type!r}")

