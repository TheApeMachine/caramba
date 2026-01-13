"""Tensor splitting operation

Divides a tensor into smaller chunks along a specified dimension.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class SplitOperation(Operation):
    """Split tensor into equal-sized chunks

    Divides input tensor into multiple smaller tensors of equal size along a dimension,
    useful for processing long sequences in smaller pieces or creating parallel computations.
    """
    def __init__(self, *, split_size: int, dim: int = -1) -> None:
        super().__init__()

        if split_size <= 0:
            raise ValueError("split_size must be > 0")

        self.split_size = split_size
        self.dim = dim

    def forward(self, *, x: Tensor) -> tuple[Tensor, ...]:
        """Divide tensor into chunks

        Splits the input tensor into multiple smaller tensors of equal size
        along the specified dimension, returning them as a tuple.
        """
        return x.split(self.split_size, dim=self.dim)
