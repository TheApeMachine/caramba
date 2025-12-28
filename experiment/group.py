"""A group of experiments that relate to each other

Grouping experiments together makes it much easier to compare
them, and it also helps the platform to understand the relationships.
"""
from __future__ import annotations

from config.manifest import Manifest
from config.group import Group
from config.train import TrainConfig

class ExperimentGroup:
    """A group of experiments that relate to each other"""
    def __init__(self, manifest: Manifest, name: str | None = None) -> None:
        self.manifest = manifest
        self.config: Group | None = None
        self.find(name)

    def find(self, name: str | None = None) -> Group:
        """Find group by name or return first group."""
        if not self.manifest.groups:
            raise ValueError("Manifest has no groups defined")

        if name is None:
            group = self.manifest.groups[0]
        else:
            group = next((
                group for group in self.manifest.groups if group.name == name
            ), None)

        if group is None:
            raise ValueError(f"Group '{name}' not found in manifest")

        self.config = group
        self.name = group.name
        self.description = group.description
        self.data = group.data
        self.runs = group.runs
        self.benchmarks = group.benchmarks

        self.validate()

        return group

    def get_train_config(self) -> TrainConfig:
        """Get train config from the first run with training."""
        for run in self.runs:
            if run.train:
                return run.train
        raise ValueError(f"Group '{self.name}' has no runs with train config")

    def validate(self) -> None:
        """Validate the group."""
        if not self.name:
            raise ValueError("Group has no name")
        if not self.description:
            raise ValueError("Group has no description")
        if not self.data:
            raise ValueError("Group has no data")
        if not self.runs:
            raise ValueError("Group has no runs")
        if not self.benchmarks:
            raise ValueError("Group has no benchmarks")