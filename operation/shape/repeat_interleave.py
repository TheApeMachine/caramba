"""Repeat-interleave operation.

This is commonly used for grouped-query attention (GQA) where K/V have fewer
heads than Q, so K/V heads are repeated to match the query head count.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class RepeatInterleaveOperation(Operation):
    """Repeat tensor elements along a dimension."""

    def __init__(self, *, repeats: int, dim: int) -> None:
        super().__init__()
        if int(repeats) <= 0:
            raise ValueError("repeats must be > 0")
        self.repeats = int(repeats)
        self.dim = int(dim)

    def forward(self, *, x: Tensor) -> Tensor:
        return x.repeat_interleave(self.repeats, dim=self.dim)

