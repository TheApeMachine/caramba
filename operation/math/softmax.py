"""Softmax activation function

Normalizes input into probability distribution.
"""
from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from caramba.operation.math.base import MathOperation


class SoftmaxOperation(MathOperation):
    """Softmax normalization

    Converts input values into a probability distribution by exponentiating
    and normalizing, ensuring outputs sum to 1. Essential for attention mechanisms.
    """
    def __init__(self, *, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *, x: Tensor) -> Tensor:
        """Apply softmax normalization

        Exponentiates input values and normalizes along specified dimension
        to create probability distribution, commonly used in attention and classification.
        """
        return F.softmax(x, dim=self.dim)