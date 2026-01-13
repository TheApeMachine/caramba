"""Transpose dimensions operation

Reorders tensor dimensions to match expected input formats for different operations.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class TransposeOperation(Operation):
    """Transpose tensor dimensions

    Swaps two dimensions of a tensor, essential for rearranging data layout
    in attention mechanisms and other neural network components.
    """
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, *, x: Tensor) -> Tensor:
        """Swap tensor dimensions

        Exchanges the positions of two dimensions, commonly used to prepare
        tensors for matrix multiplication or to match expected input shapes.
        """
        return x.transpose(self.dim0, self.dim1)