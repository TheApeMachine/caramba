"""Experiment orchestration.

The ExperimentRunner coordinates all phases of an experiment:
1. Parse and validate the manifest
2. Run training (upcycling with distillation)
3. Run benchmarks comparing teacher and student
4. Generate artifacts for analysis and publication
5. (Optional) Draft/update paper with AI agent
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch import nn

from compiler import Compiler
from config.manifest import Manifest
from console import logger
from trainer.upcycle import Upcycle

from experiment.group import ExperimentGroup
from experiment.benchmarks import ExperimentBenchmarks
from experiment.results import ExperimentResults

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
        path = Path(f"./config/presets/{self.manifest.name}.yml")
        manifest = Manifest.from_path(path)

        # Lower and validate
        compiler = Compiler()
        manifest = compiler.lowerer.lower_manifest(manifest)
        compiler.validator.validate_manifest(manifest)

        runner = ExperimentRunner(manifest)

        try:
            group = ExperimentGroup(self.manifest, group_name)
        except ValueError as e:
            logger.error(f"Error finding group: {e} - ending experiment")
            sys.exit(1)

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
