"""Experiment orchestration.

The ExperimentRunner coordinates all phases of an experiment:
1. Parse and validate the manifest
2. Run training (upcycling with distillation)
3. Run benchmarks comparing teacher and student
4. Generate artifacts for analysis and publication
5. (Optional) Draft/update paper with AI agent

This module also provides a manifest-driven *dispatcher* so the CLI can remain
a single entrypoint: run what the manifest declares (groups or agent processes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from compiler import Compiler
from config.manifest import Manifest
from console import logger
from trainer.upcycle import Upcycle

from experiment.group import ExperimentGroup
from experiment.benchmarks import ExperimentBenchmarks
from experiment.results import ExperimentResults


def _resolve_target(manifest: Manifest, target: str | None) -> str:
    """Resolve an execution target.

    Resolution order:
    - explicit CLI target, optionally via entrypoints alias
    - entrypoints['default']
    - first group
    - first agent process
    """
    if target:
        if ":" in target:
            return target
        if manifest.entrypoints and target in manifest.entrypoints:
            return manifest.entrypoints[target]
        return target

    if manifest.entrypoints and "default" in manifest.entrypoints:
        return manifest.entrypoints["default"]

    if manifest.groups:
        return f"group:{manifest.groups[0].name}"

    if manifest.agents and manifest.agents.processes:
        return f"process:{manifest.agents.processes[0].name}"

    raise ValueError("Manifest has no runnable targets (no groups and no agent processes).")


def _parse_target(target: str) -> tuple[str, str]:
    if ":" not in target:
        raise ValueError(
            f"Invalid target '{target}'. Expected 'group:<name>' or 'process:<name>'."
        )
    kind, name = target.split(":", 1)
    kind = kind.strip().lower()
    name = name.strip()
    if kind not in {"group", "process"}:
        raise ValueError(
            f"Invalid target kind '{kind}' for '{target}'. Expected 'group' or 'process'."
        )
    if not name:
        raise ValueError(f"Invalid target '{target}': missing name after ':'.")
    return kind, name


def run_from_manifest_path(
    manifest_path: Path,
    *,
    target: str | None = None,
    dry_run: bool = False,
) -> Any:
    """Single manifest-driven entrypoint for the whole platform."""
    manifest = Manifest.from_path(manifest_path)
    resolved = _resolve_target(manifest, target)
    kind, name = _parse_target(resolved)

    if kind == "group":
        if manifest.model is None:
            raise ValueError("Target is a group but manifest has no 'model' section.")
        if not manifest.groups:
            raise ValueError("Target is a group but manifest has no 'groups' section.")

        compiler = Compiler()
        lowered = compiler.lowerer.lower_manifest(manifest)
        compiler.validator.validate_manifest(lowered, print_plan=True)
        if dry_run:
            return {}
        return ExperimentRunner(lowered).run(group_name=name)

    # kind == "process"
    if manifest.agents is None:
        raise ValueError("Target is a process but manifest has no 'agents' section.")
    if dry_run:
        # Process-specific preflight is performed by the process runner.
        from agent.process_runner import (  # pyright: ignore[reportMissingImports]
            dry_run_process,
        )  # local import

        return dry_run_process(manifest, process_name=name, manifest_path=manifest_path)

    from agent.process_runner import run_process  # pyright: ignore[reportMissingImports]  # local import

    return run_process(manifest, process_name=name, manifest_path=manifest_path)


class ExperimentRunner:
    """Unified experiment runner for the complete pipeline.

    Takes a manifest and runs all configured groups through upcycling,
    benchmarking, and artifact generation.

    Usage:
        manifest = Manifest.from_path("llama32_1b_dba.yml")
        runner = ExperimentRunner(manifest)
        artifacts = runner.run()  # Returns paths to generated artifacts
    """
    def __init__(self, manifest: Manifest) -> None:
        """Initialize with a validated manifest."""
        self.manifest = manifest
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None
        self.benchmarks = ExperimentBenchmarks(manifest)
        self.results = ExperimentResults(manifest)

    def run(
        self,
        group_name: str | None = None,
        *,
        resume_from: Path | None = None,
        benchmarks_only: bool = False,
    ) -> dict[str, Path]:
        """Run the complete experiment pipeline.

        Args:
            group_name: Optional group name to run. If None, runs the first group.
            resume_from: Optional path to a checkpoint to resume from.
            benchmarks_only: If True, skip training runs and only run benchmarks.

        Returns:
            Dict mapping artifact names to their file paths.
        """
        group = ExperimentGroup(self.manifest, group_name)

        logger.header("Experiment", group.name)
        logger.info(group.description)

        logger.key_value(
            {
                "Runs": len(group.runs),
                "Benchmarks": len(group.benchmarks) if group.benchmarks else 0,
                "Data": group.data,
            }
        )

        # Get train config
        train_config = group.get_train_config()

        if group.config is None:
            raise ValueError(f"Group '{group.name}' config not loaded")

        # Run upcycle training (or load from checkpoint)
        upcycle = Upcycle(
            self.manifest,
            group.config,
            train_config,
            defaults=self.manifest.defaults,
            resume_from=resume_from,
        )

        if not benchmarks_only:
            for i, run in enumerate(group.runs):
                phase_name = run.train.phase.value if run.train else "unknown"
                logger.step(i + 1, len(group.runs), f"Run '{run.id}' ({phase_name})")
                upcycle.run(run)
        elif resume_from is None:
            logger.warning(
                "benchmarks_only=True but no --resume-from provided. "
                "Benchmarks will run with freshly initialized (untrained) weights."
            )

        # Store references to trained models
        self.teacher = upcycle.teacher
        self.student = upcycle.student

        # Run benchmarks if configured
        artifacts: dict[str, Path] = {}
        if group.benchmarks:
            artifacts = self.benchmarks.run(group, upcycle)

        logger.success(f"Experiment complete • {len(artifacts)} artifacts generated")
        return artifacts
