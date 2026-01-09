"""Checkpoint loading.

Provides checkpoint implementations for different on-disk formats, plus a small
builder that selects the correct implementation from a path. This keeps all
format knowledge localized so the rest of the system can operate on state_dicts
without caring where they came from.
"""

from __future__ import annotations

from caramba.loader.checkpoint.base import Checkpoint
from caramba.loader.checkpoint.builder import CheckpointBuilder
from caramba.loader.checkpoint.hf import HFCheckpoint
from caramba.loader.checkpoint.pytorch import CheckpointPytorch
from caramba.loader.checkpoint.safetensors import CheckpointSafetensors
from caramba.loader.checkpoint.sharded import CheckpointSharded

__all__ = [
    "Checkpoint",
    "CheckpointBuilder",
    "HFCheckpoint",
    "CheckpointPytorch",
    "CheckpointSafetensors",
    "CheckpointSharded",
]

