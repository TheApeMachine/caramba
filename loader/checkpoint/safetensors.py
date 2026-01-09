"""Safetensors checkpoint loader.

Safetensors stores tensors without pickle, enabling fast and safe loading of
model weights.
"""
from __future__ import annotations

from pathlib import Path
from safetensors.torch import load_file
from torch import Tensor

from caramba.core.platform import Platform
from caramba.loader.checkpoint.base import Checkpoint, StateDict
from caramba.loader.checkpoint.error import CheckpointError, CheckpointErrorType


class CheckpointSafetensors(Checkpoint):
    """Safetensors checkpoint loader."""
    def __init__(self, platform: Platform = Platform()) -> None:
        super().__init__(platform)

    def load(self, path: Path) -> StateDict:
        p = Path(path)

        if not p.exists():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_NOT_FOUND)
        if p.is_dir():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_IS_DIRECTORY)

        out = load_file(p, device=self.platform.device.value)

        if not isinstance(out, dict):
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_INVALID_FORMAT)

        for key, value in out.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise CheckpointError(CheckpointErrorType.CHECKPOINT_UNSUPPORTED)

        return out

