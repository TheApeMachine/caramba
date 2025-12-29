"""Small utility ops for GraphTopology nodes.

GraphTopology nodes can reference these via:

  op: python:topology.ops:Concat

This keeps common graph wiring (concat/add/etc.) declarative in manifests.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override


class Concat(nn.Module):
    """Concatenate N tensors along a dimension."""

    def __init__(self, *, dim: int = -1) -> None:
        super().__init__()
        self.dim = int(dim)

    @override
    def forward(self, *xs: Tensor, ctx: object | None = None) -> Tensor:
        _ = ctx
        if len(xs) < 1:
            raise ValueError("Concat requires at least one input tensor")
        return torch.cat(xs, dim=self.dim)


class Add(nn.Module):
    """Elementwise add of two tensors."""

    @override
    def forward(self, a: Tensor, b: Tensor, *, ctx: object | None = None) -> Tensor:
        _ = ctx
        return a + b

