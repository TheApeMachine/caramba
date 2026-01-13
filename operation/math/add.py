"""Tensor addition operation

Adds two tensors element-wise or adds a bias term.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.math.base import MathOperation


class AddOperation(MathOperation):
    """Element-wise tensor addition

    Adds two tensors element-wise or adds a bias/residual term to input,
    fundamental for residual connections and bias addition in neural networks.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, a: Tensor, b: Tensor) -> Tensor:
        """Add two tensors element-wise

        Performs element-wise addition between tensors a and b,
        commonly used for residual connections, bias addition, and skip connections.
        """
        return a + b