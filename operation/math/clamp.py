"""Tensor clamping operation

Constrains tensor values within specified range.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.math.base import MathOperation


class ClampOperation(MathOperation):
    """Constrain tensor values to range

    Clamps tensor values to lie within specified minimum and maximum bounds,
    commonly used for numerical stability and output constraints.
    """
    def __init__(self, *, min_val: float | None = None, max_val: float | None = None) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, *, x: Tensor) -> Tensor:
        """Clamp tensor values

        Constrains all elements of input tensor to lie within [min_val, max_val] range,
        useful for numerical stability and enforcing output constraints.
        """
        if self.min_val is None and self.max_val is None:
            return x  # No clamping needed
        return x.clamp(min=self.min_val, max=self.max_val)