"""Optimizer state offloading utilities.

On CUDA, optimizer offload implies PCIe transfers. On Apple Silicon (UMA), the
physical memory is shared, but PyTorch still models CPU vs MPS tensors as
distinct devices. Offloading here is therefore best-effort: it can reduce MPS
allocations at the cost of device transfers.

This module provides a simple policy hook for trainers.
"""

from __future__ import annotations

import torch


def _move_state_tensor(v: object, device: torch.device) -> object:
    if isinstance(v, torch.Tensor):
        return v.to(device=device)
    return v


def offload_optimizer_state(optimizer: torch.optim.Optimizer, *, device: torch.device | None = None) -> None:
    """Move optimizer state tensors to CPU (or another device)."""
    target = torch.device("cpu") if device is None else torch.device(device)
    for _p, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        for k, v in list(state.items()):
            state[k] = _move_state_tensor(v, target)


def load_optimizer_state(optimizer: torch.optim.Optimizer, *, device: torch.device) -> None:
    """Move optimizer state tensors to the given device."""
    target = torch.device(device)
    for _p, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        for k, v in list(state.items()):
            state[k] = _move_state_tensor(v, target)

