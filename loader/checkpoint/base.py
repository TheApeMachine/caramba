"""Checkpoint base.

Defines the checkpoint abstraction used throughout the loader system to turn
on-disk artifacts into in-memory state_dict mappings. Keeping this interface
stable lets us add formats without rippling changes across the codebase.
"""

from __future__ import annotations

import abc
from pathlib import Path

from torch import Tensor

from core.platform import Platform

StateDict = dict[str, Tensor]


class Checkpoint(abc.ABC):
    """Checkpoint interface.

    A checkpoint is a source of a PyTorch-style state_dict: a mapping from
    parameter names to tensors.
    """
    # These attributes are set in __init__, but declared here for type checking
    platform: Platform
    base_path: Path

    def __init__(self, platform: Platform) -> None:
        self.base_path = Path("artifacts/checkpoints")
        self.platform = platform

    @abc.abstractmethod
    def load(self, path: Path) -> StateDict:
        """Load a state_dict from `path`."""


