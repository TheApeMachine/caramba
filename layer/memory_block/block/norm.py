"""RMS normalization

Provides a small, composable RMSNorm-style normalization used inside MOSAIC blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class RmsNorm:
    """RMS normalization.

    Normalization keeps token activations in a stable scale regime, improving
    numerical stability and gating behavior in streaming architectures.
    """

    eps: float = 1e-6

    def apply(self, x: Tensor) -> Tensor:
        if x.ndim < 1:
            raise ValueError("x must have at least 1 dimension")
        return x * (x.pow(2).mean(dim=-1, keepdim=True) + float(self.eps)).rsqrt()

