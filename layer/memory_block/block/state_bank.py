"""Multiscale state bank

Implements a bank of leaky integrators with learnable decay rates. This provides
long-horizon intent tracking without attention.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from caramba.carmath import leaky_integrator_scan


class StateBank(nn.Module):
    """Multiscale state bank

    A bank of leaky integrators provides multiple “time constants” in parallel,
    so some state tracks short-term detail while other state changes slowly and
    can carry longer-horizon intent.
    """

    def __init__(self, *, state_k: int, state_in: nn.Linear, state_out: nn.Linear, decay_logit: nn.Parameter) -> None:
        super().__init__()
        if int(state_k) < 1:
            raise ValueError("state_k must be >= 1")
        self.state_k = int(state_k)
        self.state_in = state_in
        self.state_out = state_out
        self.decay_logit = decay_logit

    def decay(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        d = torch.sigmoid(self.decay_logit).to(dtype=dtype, device=device)
        return d.view(1, int(self.state_k), 1)

    def scan(self, u: Tensor, *, s0: Tensor) -> tuple[Tensor, Tensor]:
        """Scan a chunk of tokens

        Scanning lets you update the state across a whole sequence chunk with a
        vectorized operator, which is much faster than a Python loop.
        """
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, D = u.shape
        inp = self.state_in(u).view(B, T, int(self.state_k), D).permute(0, 2, 1, 3)
        s_seq, s_last = leaky_integrator_scan(inp, s0, self.decay(dtype=u.dtype, device=u.device))
        g_in = s_seq.to(dtype=u.dtype).permute(0, 2, 1, 3).reshape(B, T, int(self.state_k) * D)
        g = self.state_out(g_in)
        return g, s_last.to(dtype=u.dtype)

    def step(self, u_t: Tensor, *, s: Tensor) -> tuple[Tensor, Tensor]:
        """Update state for one token

        This is the “true” recurrent update used during decoding, where you only
        have one new token at a time.
        """
        if u_t.ndim != 2:
            raise ValueError(f"u_t must have shape (B,D), got {tuple(u_t.shape)}")
        B, D = u_t.shape
        decay = torch.sigmoid(self.decay_logit).to(dtype=u_t.dtype, device=u_t.device).view(1, int(self.state_k), 1)
        inp = self.state_in(u_t).view(B, int(self.state_k), D)
        s_next = decay * s + inp
        g_t = self.state_out(s_next.reshape(B, int(self.state_k) * D))
        return g_t, s_next

