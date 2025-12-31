"""Optimization-related math helpers."""

from __future__ import annotations

import math

import torch
from torch import nn

from console import logger


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


def safe_perplexity_from_nll(nll: float, *, max_nll: float = 20.0) -> float:
    """Convert NLL to perplexity with overflow/NaN guards.

    This is a small but frequently repeated piece of training math. Keeping it
    here avoids duplicated "if finite, exp, else inf" logic in trainers.
    """
    try:
        x = float(nll)
        if not math.isfinite(x):
            return float("inf")
        if x > float(max_nll):
            return float("inf")
        return float(math.exp(x))
    except Exception as e:
        logger.warning(f"Failed to convert NLL to perplexity: {e}")
        return float("inf")

