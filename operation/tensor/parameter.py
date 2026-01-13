"""Learnable parameter tensor operation.

This provides a manifest-friendly way to introduce learned tensors into an
op-graph (e.g. DBA null-attention keys/values) without needing bespoke layer
classes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from caramba.operation.base import Operation

_Init = Literal["zeros", "ones", "normal", "uniform"]


class ParameterOperation(Operation):
    """Return a learned tensor parameter (optionally cast/expanded like an input)."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        init: _Init = "normal",
        normal_std: float = 0.02,
        uniform_a: float = -0.02,
        uniform_b: float = 0.02,
        requires_grad: bool = True,
        expand_batch: bool = False,
    ) -> None:
        super().__init__()

        if not shape:
            raise ValueError("shape must be non-empty")
        sizes = tuple(int(s) for s in shape)
        if any(s <= 0 for s in sizes):
            raise ValueError("shape must contain only positive ints")

        self.shape = sizes
        self.init = init
        self.normal_std = float(normal_std)
        self.uniform_a = float(uniform_a)
        self.uniform_b = float(uniform_b)
        self.requires_grad = bool(requires_grad)
        self.expand_batch = bool(expand_batch)

        self.param = nn.Parameter(torch.empty(*sizes), requires_grad=bool(requires_grad))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init = str(self.init)
        if init == "zeros":
            nn.init.zeros_(self.param)
            return
        if init == "ones":
            nn.init.ones_(self.param)
            return
        if init == "normal":
            nn.init.normal_(self.param, mean=0.0, std=float(self.normal_std))
            return
        if init == "uniform":
            nn.init.uniform_(self.param, a=float(self.uniform_a), b=float(self.uniform_b))
            return
        raise ValueError(f"Unknown init={self.init!r}")

    def forward(self, x: Tensor | None = None, *, like: Tensor | None = None) -> Tensor:  # type: ignore[override]
        ref = like if like is not None else x
        out: Tensor = self.param

        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)

        if self.expand_batch and ref is not None:
            if ref.dim() < 1:
                raise ValueError("expand_batch requires ref to be at least rank-1")
            batch = int(ref.size(0))
            if out.dim() < 1:
                raise ValueError("expand_batch requires parameter to be at least rank-1")
            if int(out.size(0)) == batch:
                return out
            if int(out.size(0)) != 1:
                raise ValueError(f"expand_batch requires parameter dim0==1 or dim0==batch ({batch}), got {out.size(0)}")
            out = out.expand(batch, *out.shape[1:])

        return out

