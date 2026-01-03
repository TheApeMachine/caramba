"""Grouping / reduction helpers (named techniques).

This module exists to centralize common "group-by key" patterns implemented
with tensor ops, so higher-level code can stay readable.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "last_write_wins",
]


def last_write_wins(keys: Tensor, times: Tensor, *, big: int | None = None) -> Tensor:
    """Return indices selecting the last event per unique key.

    Args:
      keys:  (N,) int64-ish tensor (unique-id per event)
      times: (N,) int64-ish tensor (monotone "time"; larger means later)
      big: optional base used for mixed radix sorting.

    Returns:
      winner: (M,) long tensor of indices into the original event arrays,
        one per unique key, selecting the event with the largest time.
        Ties are broken by the original event order (later index wins).
    """
    if keys.dim() != 1 or times.dim() != 1:
        raise ValueError("last_write_wins expects 1D keys and times")
    if int(keys.numel()) != int(times.numel()):
        raise ValueError("last_write_wins expects keys and times to have equal length")

    N = int(keys.numel())
    if N == 0:
        return keys.new_empty((0,), dtype=torch.long)

    k = keys.to(dtype=torch.long)
    t = times.to(dtype=torch.long)
    idx = torch.arange(N, device=k.device, dtype=torch.long)

    # Use stable multi-key sort to avoid int64 overflow: sort by (k, t, idx)
    # Sort in reverse order of priority (stable sort preserves previous order)
    # First by idx (tie-breaker), then by t, then by k
    order = torch.argsort(idx, stable=True)
    t_sorted = t[order]
    order = order[torch.argsort(t_sorted, stable=True)]
    k_sorted = k[order]
    order = order[torch.argsort(k_sorted, stable=True)]

    k_sorted = k[order]
    boundary = torch.ones((N,), device=k.device, dtype=torch.bool)
    boundary[1:] = k_sorted[1:] != k_sorted[:-1]
    starts = torch.nonzero(boundary, as_tuple=False).squeeze(-1)
    ends = torch.cat([starts[1:], torch.tensor([N], device=k.device, dtype=torch.long)]) - 1
    winner = order[ends]
    return winner

