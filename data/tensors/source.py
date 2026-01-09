"""Tensor source protocol

Defines the interface for loading tensors from different file formats,
allowing datasets to work with NumPy, safetensors, or other sources
through a unified interface.
"""
from __future__ import annotations

from typing import Protocol

from torch import Tensor


class TensorSource(Protocol):
    """Tensor source protocol

    Defines the interface for loading tensors from different file formats,
    allowing the dataset to work with NumPy, safetensors, or other sources
    through a unified interface.
    """
    def __len__(self) -> int: ...
    def get(self, idx: int) -> Tensor: ...
