"""Optimization-related math helpers."""

from __future__ import annotations

import math

import torch
from torch import nn

from caramba.console import logger


def global_grad_norm_l2(model: nn.Module) -> float:
    """Compute global L2 norm of gradients across all parameters.

    Performance note:
    - Avoids per-parameter `.item()` syncs; only syncs once at the end.
    - Uses foreach ops to reduce kernel-launch overhead on accelerators.
    """
    grads: list[torch.Tensor] = []
    for p in model.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        grads.append(g.detach())
    if not grads:
        return 0.0

    if not hasattr(torch, "_foreach_norm"):
        raise RuntimeError(
            "global_grad_norm_l2 requires torch._foreach_norm to be available.\n"
            "Fix: upgrade to a PyTorch build that includes foreach ops."
        )
    norms = torch._foreach_norm(grads)  # type: ignore[attr-defined]
    # Accumulate squared norms in float32.
    ns = torch.stack([n.to(dtype=torch.float32) for n in norms], dim=0)
    total = (ns * ns).sum().sqrt()
    return float(total.item())


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
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to convert NLL to perplexity: {e}")
        return float("inf")

