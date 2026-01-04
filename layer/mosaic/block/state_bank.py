"""Multiscale state bank

Implements a bank of leaky integrators with learnable decay rates. This provides
long-horizon intent tracking without attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from caramba.carmath import leaky_integrator_scan


@dataclass(slots=True)
class StateBank:
    """State bank."""

    state_k: int
    state_in: nn.Linear
    state_out: nn.Linear
    decay_logit: nn.Parameter

    def decay(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        d = torch.sigmoid(self.decay_logit).to(dtype=dtype, device=device)
        return d.view(1, int(self.state_k), 1)

    def scan(self, u: Tensor, *, s0: Tensor) -> tuple[Tensor, Tensor]:
        """Chunk scan: returns (g_seq, s_last)."""
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, D = u.shape
        inp = self.state_in(u).view(B, T, int(self.state_k), D).permute(0, 2, 1, 3)
        s_seq, s_last = leaky_integrator_scan(inp, s0, self.decay(dtype=u.dtype, device=u.device))
        g_in = s_seq.to(dtype=u.dtype).permute(0, 2, 1, 3).reshape(B, T, int(self.state_k) * D)
        g = self.state_out(g_in)
        return g, s_last.to(dtype=u.dtype)

    def step(self, u_t: Tensor, *, s: Tensor) -> tuple[Tensor, Tensor]:
        """Streaming update for one token: returns (g_t, s_next)."""
        if u_t.ndim != 2:
            raise ValueError(f"u_t must have shape (B,D), got {tuple(u_t.shape)}")
        B, D = u_t.shape
        decay = torch.sigmoid(self.decay_logit).to(dtype=u_t.dtype, device=u_t.device).view(1, int(self.state_k), 1)
        inp = self.state_in(u_t).view(B, int(self.state_k), D)
        s_next = decay * s + inp
        g_t = self.state_out(s_next.reshape(B, int(self.state_k) * D))
        return g_t, s_next

