"""Checkpoint builder.

Selects an appropriate checkpoint loader implementation based on the path.
This keeps format detection centralized and manifest-driven.
"""

from __future__ import annotations

from pathlib import Path

from caramba.core.platform import Platform
from caramba.loader.checkpoint.base import Checkpoint
from caramba.loader.checkpoint.base import StateDict
from caramba.loader.checkpoint.pytorch import CheckpointPytorch
from caramba.loader.checkpoint.safetensors import CheckpointSafetensors
from caramba.loader.checkpoint.sharded import CheckpointSharded


class CheckpointBuilder:
    """Build checkpoint loaders from a path."""

    def build(self, path: Path, platform: Platform = Platform()) -> Checkpoint:
        p = Path(path)

        if p.name.endswith(".index.json"):
            return CheckpointSharded(builder=self)

        if p.suffix == ".safetensors":
            return CheckpointSafetensors(platform)

        return CheckpointPytorch(platform)

    def load(self, path: Path) -> StateDict:
        """Load a checkpoint as a state_dict."""
        return self.build(path).load(path)

