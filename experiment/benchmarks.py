"""Experiment benchmarking orchestration.

Handles setting up and running benchmarks for an experiment group.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.runner import BenchmarkRunner
from caramba.config.benchmark import BenchmarkSuite
from caramba.console import logger

if TYPE_CHECKING:
    from caramba.experiment.group import ExperimentGroup
    from caramba.config.manifest import Manifest

class ExperimentBenchmarks:
    """Orchestrates benchmarks for an experiment."""
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def run(self, group: ExperimentGroup, upcycle: Any) -> dict[str, Path]:
        """Run benchmarks and generate artifacts."""
        if not group.benchmarks:
            return {}

        # Get output formats from manifest, with default fallback
        default_formats = ["csv", "json", "png", "latex"]
        output_formats = getattr(self.manifest, "output_formats", None)
        if (
            output_formats is None
            or not isinstance(output_formats, list)
            or not output_formats
            or not all(isinstance(f, str) for f in output_formats)
        ):
            if output_formats is not None:
                logger.warning(
                    "Invalid or empty output_formats in manifest, using defaults"
                )
            output_formats = list(default_formats)

        # Build benchmark suite
        suite = BenchmarkSuite(
            benchmarks=group.benchmarks,
            output_dir=f"artifacts/{self.manifest.name or 'experiment'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=output_formats,
        )

        # Build metadata
        train_config = group.get_train_config()
        model = getattr(self.manifest, "model", None)
        topology = getattr(model, "topology", None) if model is not None else None
        topology_type = (
            str(getattr(topology, "type", "")) if topology is not None else ""
        )

        metadata = ExperimentMetadata(
            name=self.manifest.name or "experiment",
            timestamp=datetime.now().isoformat(),
            manifest_path=str(self.manifest.name) if self.manifest.name else "",
            teacher_checkpoint=train_config.teacher_ckpt or "",
            student_config=topology_type,
            device=train_config.device,
            notes=self.manifest.notes,
        )

        # Run benchmarks
        runner = BenchmarkRunner(suite, upcycle.device, metadata)
        return runner.run(upcycle.teacher, upcycle.student)
