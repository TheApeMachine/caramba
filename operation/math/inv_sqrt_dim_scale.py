"""Scale by inverse sqrt of last dimension.

This is a small helper used in attention-style computations where logits are
scaled by 1/sqrt(d). In op graphs, it avoids hard-coding head-dim-derived
constants in YAML.
"""
from __future__ import annotations

import math

from torch import Tensor

from caramba.operation.math.base import MathOperation


class InvSqrtDimScaleOperation(MathOperation):
    """Scale a tensor by 1/sqrt(x.size(-1))."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, x: Tensor) -> Tensor:
        d = int(x.size(-1))
        if d <= 0:
            raise ValueError("Expected x.size(-1) > 0")
        return x * (1.0 / math.sqrt(float(d)))

