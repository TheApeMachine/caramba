"""Experiment results processing.

Handles building summaries and reports from experiment artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from caramba.console import logger

if TYPE_CHECKING:
    from caramba.experiment.group import ExperimentGroup
    from caramba.config.manifest import Manifest

class ExperimentResults:
    """Builds summaries of experiment results."""
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def build_summary(
        self,
        group: ExperimentGroup,
        artifacts: dict[str, Path],
    ) -> dict:
        """Build a summary of experiment results for the paper drafter."""
        results = {
            "experiment_name": self.manifest.name,
            "group_name": group.name,
            "group_description": group.description,
            "notes": self.manifest.notes,
            "runs": [],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }

        # Add run information
        for run in group.runs:
            run_info = {
                "id": run.id,
                "mode": run.mode,
                "seed": run.seed,
                "steps": run.steps,
            }
            if run.train:
                run_info["train"] = {
                    "phase": run.train.phase.value if run.train.phase else None,
                    "batch_size": run.train.batch_size,
                    "block_size": run.train.block_size,
                    "lr": run.train.lr,
                    "device": run.train.device,
                }
            results["runs"].append(run_info)

        # Try to load report.json if it exists
        for name, path in artifacts.items():
            if name == "report.json" and path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        report = json.load(f)
                        results["benchmark_summary"] = report.get("summary", {})
                        results["metadata"] = report.get("metadata", {})
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load report.json from {path}: {e}")

        return results
