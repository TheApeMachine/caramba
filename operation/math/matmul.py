"""Matrix multiplication operation

Core computational primitive for neural networks and attention mechanisms.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.math.base import MathOperation


class MatMulOperation(MathOperation):
    """Matrix multiplication between tensors

    Performs matrix multiplication between two tensors, fundamental operation
    used in linear layers, attention mechanisms, and most neural network computations.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, a: Tensor, b: Tensor) -> Tensor:
        """Multiply two matrices

        Computes matrix multiplication between input tensors a and b,
        following standard matrix multiplication rules for compatible dimensions.
        """
        return torch.matmul(a, b)