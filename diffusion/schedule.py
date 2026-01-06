"""Diffusion schedules

Provides manifest-friendly noise schedules and shaping utilities for diffusion
training and sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class NoiseSchedule:
    """Noise schedule for diffusion

    Produces alpha_bar (alphas_cumprod) used by both training and samplers.
    """

    kind: str
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def alphasCumprod(self, *, timesteps: int, device: torch.device) -> Tensor:
        """Compute alpha_bar over timesteps on device."""

        kind = str(self.kind).lower().strip()
        if kind == "linear":
            betas = torch.linspace(
                float(self.beta_start),
                float(self.beta_end),
                int(timesteps),
                device=device,
            )
        elif kind == "cosine":
            betas = self.cosineBetas(timesteps=int(timesteps), device=device)
        else:
            raise ValueError(f"Unknown noise schedule kind: {self.kind!r}")

        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0)

    def cosineBetas(self, *, timesteps: int, device: torch.device) -> Tensor:
        """Cosine beta schedule (Nichol & Dhariwal)."""

        s = 0.008
        steps = torch.arange(int(timesteps) + 1, device=device) / int(timesteps)
        alpha = torch.cos((steps + s) / (1.0 + s) * torch.pi / 2.0) ** 2
        alpha = alpha / alpha[0]
        betas = 1.0 - (alpha[1:] / alpha[:-1])
        return betas.clamp(1e-5, 0.999)

    def gatherToShape(self, *, values: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather values[t] and broadcast to `shape`."""

        if t.dim() != 1:
            raise ValueError(f"Expected t to be 1D, got shape={tuple(t.shape)}")
        batch = int(t.shape[0])
        gathered = values.gather(0, t)
        view = gathered.view(batch, *((1,) * (len(shape) - 1)))
        return view.expand(shape)

