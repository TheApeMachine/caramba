"""Element-wise tensor multiplication operation

Multiplies two tensors element-wise.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.math.base import MathOperation


class MulOperation(MathOperation):
    """Element-wise tensor multiplication

    Multiplies two tensors element-wise, used for gating mechanisms,
    attention weights application, and other multiplicative operations.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, a: Tensor, b: Tensor) -> Tensor:
        """Multiply tensors element-wise

        Performs element-wise multiplication between tensors a and b,
        preserving tensor shape while combining corresponding elements.
        """
        return a * b