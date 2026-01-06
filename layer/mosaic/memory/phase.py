"""Phase scoring for MOSAIC memory

Adds an optional phase-resonant similarity channel for in-bucket selection.
This implements global-phase-invariant similarity via the magnitude of the
complex inner product over phasor-coded vectors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PhaseTagProjector(nn.Module):
    """Phase tag projector

    Maps real-valued vectors to bounded per-dimension angles in [-pi, pi] via a
    learned linear projection followed by tanh scaling.
    """

    def __init__(self, *, in_dim: int, phase_dim: int, tanh_scale: float) -> None:
        super().__init__()
        if int(in_dim) < 1:
            raise ValueError("in_dim must be >= 1")
        if int(phase_dim) < 1:
            raise ValueError("phase_dim must be >= 1")
        if float(tanh_scale) <= 0.0:
            raise ValueError("tanh_scale must be > 0")
        self.in_dim = int(in_dim)
        self.phase_dim = int(phase_dim)
        self.tanh_scale = float(tanh_scale)
        self.proj = nn.Linear(int(in_dim), int(phase_dim), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if int(x.size(-1)) != int(self.in_dim):
            raise ValueError(
                f"x last dimension must match in_dim={int(self.in_dim)}, got {int(x.size(-1))}"
            )
        z = self.proj(x)
        angles = torch.tanh(z * float(self.tanh_scale)) * float(math.pi)
        return angles.to(dtype=x.dtype)


class PhaseSimilarity(nn.Module):
    """Phase similarity

    Phase similarity measures alignment in a way that is invariant to a global
    phase shift, which makes it behave like a robust “direction match” even when
    individual components wrap around at ±pi.
    """

    def __init__(self, *, phase_dim: int) -> None:
        super().__init__()
        if int(phase_dim) < 1:
            raise ValueError("phase_dim must be >= 1")
        self.phase_dim = int(phase_dim)

    def score(self, *, q_angles: Tensor, k_angles: Tensor, valid: Tensor, batch: int, time: int) -> Tensor:
        if q_angles.ndim != 3:
            raise ValueError(f"q_angles must be (B,T,N), got {tuple(q_angles.shape)}")
        if k_angles.ndim != 5:
            raise ValueError(f"k_angles must be (B,H,T,A,N), got {tuple(k_angles.shape)}")
        if int(q_angles.size(-1)) != int(self.phase_dim):
            raise ValueError("q_angles last dim must match phase_dim")
        if int(k_angles.size(-1)) != int(self.phase_dim):
            raise ValueError("k_angles last dim must match phase_dim")

        cq = torch.cos(q_angles).view(batch, 1, time, 1, self.phase_dim)
        sq = torch.sin(q_angles).view(batch, 1, time, 1, self.phase_dim)
        ck = torch.cos(k_angles)
        sk = torch.sin(k_angles)

        real = (cq * ck + sq * sk).sum(dim=-1)
        imag = (sq * ck - cq * sk).sum(dim=-1)
        mag = torch.sqrt(real * real + imag * imag) * (1.0 / float(self.phase_dim))
        return mag.masked_fill(~valid, float("-inf"))

