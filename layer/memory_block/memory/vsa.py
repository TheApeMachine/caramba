"""VSA helpers for MOSAIC memory

Adds a low-cost vector-symbolic channel to improve robustness of in-bucket
selection and to provide a novelty signal for sparse-write dynamics, without
changing hard bucket routing or introducing global attention.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VsaTagProjector(nn.Module):
    """Fixed random projection into a VSA-like tag space.

    Uses a Rademacher (±1) matrix and a tanh squash to produce near-±1 vectors.
    This is cheap, differentiable, and works well with Hadamard/role-filler VSA
    later without committing to a specific binding operator yet.
    """

    def __init__(self, *, in_dim: int, vsa_dim: int, tanh_scale: float) -> None:
        super().__init__()
        if int(in_dim) < 1:
            raise ValueError("in_dim must be >= 1")
        if int(vsa_dim) < 1:
            raise ValueError("vsa_dim must be >= 1")
        if float(tanh_scale) <= 0.0:
            raise ValueError("tanh_scale must be > 0")

        self.in_dim = int(in_dim)
        self.vsa_dim = int(vsa_dim)
        self.tanh_scale = float(tanh_scale)

        w = torch.empty((self.in_dim, self.vsa_dim), dtype=torch.float32)
        w.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        w.mul_(1.0 / math.sqrt(float(self.in_dim)))
        self.register_buffer("proj_weight", w, persistent=True)
        self.proj_weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if int(x.size(-1)) != int(self.in_dim):
            raise ValueError(
                f"x last dimension must match in_dim={int(self.in_dim)}, got {int(x.size(-1))}"
            )
        y = torch.matmul(x, self.proj_weight.to(dtype=x.dtype, device=x.device))
        y = torch.tanh(y * float(self.tanh_scale))
        n = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(1e-6)
        return y / n


class VsaNovelty(nn.Module):
    """Novelty factor from VSA similarity.

    Produces a soft novelty in [0,1] from a max similarity score:
    - novelty≈1 when max_sim << threshold (novel)
    - novelty≈0 when max_sim >> threshold (redundant)
    """

    def __init__(self, *, beta: float, threshold: float) -> None:
        super().__init__()
        if float(beta) <= 0.0:
            raise ValueError("beta must be > 0")
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, max_sim: Tensor) -> Tensor:
        if not isinstance(max_sim, Tensor):
            raise TypeError(f"max_sim must be a Tensor, got {type(max_sim).__name__}")
        x = (max_sim - float(self.threshold)) * float(self.beta)
        novelty = 1.0 - torch.sigmoid(x)
        return novelty.clamp(0.0, 1.0).to(dtype=max_sim.dtype)

