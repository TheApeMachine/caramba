"""Experiment target wrapper.

This module used to wrap legacy `manifest.groups[]`. The manifest schema is now
target-based, so this wrapper selects an `experiment` target and exposes a
stable, convenient view for downstream tooling.
"""
from __future__ import annotations

from config.manifest import Manifest
from config.target import ExperimentTargetConfig
from config.train import TrainConfig


class ExperimentGroup:
    """A selected experiment target with convenient accessors."""

    def __init__(self, manifest: Manifest, name: str | None = None) -> None:
        self.manifest = manifest
        self.find(name)

    def find(self, name: str | None = None) -> ExperimentTargetConfig:
        """Find an experiment target by name or return the first experiment."""
        if not self.manifest.targets:
            raise ValueError("Manifest has no targets defined")

        experiments = [t for t in self.manifest.targets if isinstance(t, ExperimentTargetConfig)]
        if not experiments:
            raise ValueError("Manifest has no experiment targets defined")

        if name is None:
            target = experiments[0]
        else:
            target = next((t for t in experiments if t.name == name), None)
            if target is None:
                raise ValueError(f"Experiment target '{name}' not found in manifest")

        self.target = target
        self.name = target.name
        self.description = getattr(target, "description", "") or ""
        self.runs = list(target.runs)
        self.benchmarks = target.benchmarks

        # Best-effort "data" string for legacy benchmarking code paths.
        p = target.data.config.get("path", None)
        self.data = str(p) if p is not None else str(target.data.ref)

        self.validate()
        return target

    def get_train_config(self) -> TrainConfig:
        """Get train config from the first run with training."""
        for run in self.runs:
            if run.train:
                return run.train
        raise ValueError(f"Experiment target '{self.name}' has no runs with train config")

    def validate(self) -> None:
        """Validate the experiment target."""
        if not self.name:
            raise ValueError("Experiment target has no name")
        if not self.description:
            # Description is optional; keep defaults quiet.
            pass
        if not self.data:
            raise ValueError("Experiment target has no data")
        if not self.runs:
            raise ValueError("Experiment target has no runs")