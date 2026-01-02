"""Scan-style tensor primitives (named techniques).

This module exists to keep raw scan math out of higher-level model code.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "leaky_integrator_scan",
]


def leaky_integrator_scan(
    inp: Tensor,
    s0: Tensor,
    decay: Tensor,
    *,
    clamp_min: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    r"""Vectorized scan for a leaky integrator bank.

    Computes:
      s_t = decay * s_{t-1} + inp_t

    Args:
      inp:  (B, K, T, D)
      s0:   (B, K, D)
      decay: broadcastable to (1, K, 1, 1). Common shapes:
        - (K,)
        - (1, K, 1)
        - (1, K, 1, 1)

    Returns:
      s_seq:  (B, K, T, D) float32 scan states
      s_last: (B, K, D) float32 last state

    Notes:
      Uses a closed form that is efficient for moderate T:
        s_t = decay^(t+1) * s0 + decay^t * cumsum_i(inp_i * decay^(-i)).
    """
    if inp.dim() != 4:
        raise ValueError("leaky_integrator_scan expects inp of shape (B,K,T,D)")
    if s0.dim() != 3:
        raise ValueError("leaky_integrator_scan expects s0 of shape (B,K,D)")

    B, K, T, D = int(inp.size(0)), int(inp.size(1)), int(inp.size(2)), int(inp.size(3))
    if T <= 0:
        empty = inp.new_zeros((B, K, 0, D), dtype=torch.float32)
        s_last = s0.to(dtype=torch.float32)
        return empty, s_last

    # Normalize decay shape.
    d = decay.to(dtype=torch.float32, device=inp.device)
    if d.dim() == 1:
        d = d.view(1, K, 1, 1)
    elif d.dim() == 3:
        d = d.view(1, K, 1, 1)
    elif d.dim() == 4:
        # Assume broadcastable already.
        pass
    else:
        raise ValueError("leaky_integrator_scan expects decay with dim in {1,3,4}")

    d = d.clamp_min(float(clamp_min))

    inp_f = inp.to(dtype=torch.float32)
    s0_f = s0.to(dtype=torch.float32)

    i = torch.arange(T, device=inp.device, dtype=torch.float32).view(1, 1, T, 1)  # (1,1,T,1)
    pow0 = d**i
    pow1 = d ** (i + 1.0)
    inv = d**(-i)
    cs = (inp_f * inv).cumsum(dim=2)
    s_seq = s0_f.unsqueeze(2) * pow1 + cs * pow0
    s_last = s_seq[:, :, -1, :]
    return s_seq, s_last

