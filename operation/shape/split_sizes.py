"""Tensor splitting by explicit section sizes.

This is the "packed projection" companion to `SplitOperation`: many transformer
blocks project multiple tensors at once (e.g. packed QKV) and then split the
result into named parts with explicit sizes.
"""
from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from caramba.operation.base import Operation


class SplitSizesOperation(Operation):
    """Split a tensor into explicit sized sections along a dimension."""

    def __init__(self, *, split_sizes: Sequence[int], dim: int = -1) -> None:
        super().__init__()

        if not split_sizes:
            raise ValueError("split_sizes must be non-empty")

        sizes = [int(s) for s in split_sizes]
        if any(s <= 0 for s in sizes):
            raise ValueError("split_sizes must contain only positive ints")

        self.split_sizes = tuple(sizes)
        self.dim = int(dim)

    def forward(self, *, x: Tensor) -> tuple[Tensor, ...]:
        dim = int(self.dim)
        nd = int(x.dim())
        dim0 = dim if dim >= 0 else nd + dim
        if dim0 < 0 or dim0 >= nd:
            raise ValueError(f"Invalid dim={dim} for tensor rank {nd}")

        expected = int(x.size(dim0))
        actual = int(sum(self.split_sizes))
        if actual != expected:
            raise ValueError(f"split_sizes sum ({actual}) must match x.size(dim) ({expected})")

        return tuple(x.split(self.split_sizes, dim=self.dim))

