"""Optimization-related math helpers."""

from __future__ import annotations

import torch
from torch import nn


def global_grad_norm_l2(model: nn.Module) -> float:
    """Compute global L2 norm of gradients across all parameters.

    Performance note:
    - Avoids per-parameter `.item()` syncs; only syncs once at the end.
    """
    total_sq: torch.Tensor | None = None
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        v = g.float().pow(2).sum()
        total_sq = v if total_sq is None else (total_sq + v)
    if total_sq is None:
        return 0.0
    return float(total_sq.sqrt().item())

