"""Checkpoint loading.

Provides checkpoint implementations for different on-disk formats, plus a small
builder that selects the correct implementation from a path. This keeps all
format knowledge localized so the rest of the system can operate on state_dicts
without caring where they came from.
"""

from __future__ import annotations

from loader.checkpoint.base import Checkpoint
from loader.checkpoint.builder import CheckpointBuilder
from loader.checkpoint.hf import HFCheckpoint
from loader.checkpoint.pytorch import CheckpointPytorch
from loader.checkpoint.safetensors import CheckpointSafetensors
from loader.checkpoint.sharded import CheckpointSharded

__all__ = [
    "Checkpoint",
    "CheckpointBuilder",
    "HFCheckpoint",
    "CheckpointPytorch",
    "CheckpointSafetensors",
    "CheckpointSharded",
]

