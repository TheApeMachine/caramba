"""PyTorch checkpoint loader.

Loads a single-file PyTorch checkpoint (e.g. `.pt` / `.bin`) into a state_dict.
This is intentionally strict: we require a safe weights-only load to avoid
pickle execution risks.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from core.platform import Platform
from loader.checkpoint.base import Checkpoint, StateDict
from loader.checkpoint.error import CheckpointError, CheckpointErrorType


class CheckpointPytorch(Checkpoint):
    """PyTorch checkpoint loader.

    Uses `weights_only=True` which requires a sufficiently new PyTorch. If the
    runtime does not support weights-only loading, we raise an actionable error
    telling the user to upgrade rather than silently falling back to unsafe
    pickle loading.
    """
    def __init__(self, platform: Platform = Platform()) -> None:
        super().__init__(platform)

    def load(self, path: Path) -> StateDict:
        p = Path(path)

        if not p.exists():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_NOT_FOUND)

        if p.is_dir():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_IS_DIRECTORY)

        if self.platform.torch_version() < (2, 4):
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_UNSUPPORTED)

        obj = torch.load(p, map_location=self.platform.device.value, weights_only=True)

        if not isinstance(obj, dict):
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_INVALID_FORMAT)

        for key, value in obj.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise CheckpointError(CheckpointErrorType.CHECKPOINT_UNSUPPORTED)

        return obj
