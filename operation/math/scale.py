"""Tensor scaling operation

Multiplies tensor by a scalar value or another tensor.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.math.base import MathOperation


class ScaleOperation(MathOperation):
    """Scale tensor by scalar or element-wise multiplication

    Multiplies input tensor by a scalar value, commonly used for temperature scaling,
    learning rate application, or element-wise multiplication in attention mechanisms.
    """
    def __init__(self, *, scale: float | Tensor = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, *, x: Tensor) -> Tensor:
        """Scale input tensor

        Multiplies the input tensor by the configured scale factor,
        supporting both scalar multiplication and element-wise tensor multiplication.
        """
        return x * self.scale