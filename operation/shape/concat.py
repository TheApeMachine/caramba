"""Tensor concatenation operation

Joins multiple tensors together along a specified dimension.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from caramba.operation.base import Operation


class ConcatOperation(Operation):
    """Concatenate tensors along a dimension

    Joins multiple input tensors into a single tensor by concatenating them
    along a specified dimension, commonly used to combine features or sequences.
    """
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *xs: Tensor) -> Tensor:
        """Join tensors together

        Concatenates multiple input tensors along the specified dimension,
        preserving all other dimensions while extending the target dimension.
        """
        return torch.cat(xs, dim=self.dim)