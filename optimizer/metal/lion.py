"""Fused Lion optimizer update wrapper for the Metal extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


def metal_lion_available() -> bool:
    return metal_supported()


def lion_fp16(
    *,
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    lr: float,
    beta1: float,
    weight_decay: float = 0.0,
    verbose_build: bool = False,
) -> Tensor:
    if p.device.type != "mps":
        raise RuntimeError("Metal Lion requires device.type == 'mps'")
    if p.dtype not in (torch.float16, torch.float32) or grad.dtype != p.dtype or m.dtype != p.dtype:
        raise RuntimeError("Metal Lion currently supports fp16/fp32 tensors only (matching)")
    if p.shape != grad.shape or p.shape != m.shape:
        raise RuntimeError(
            f"Metal Lion requires matching shapes for p/grad/m, got p={tuple(p.shape)}, grad={tuple(grad.shape)}, m={tuple(m.shape)}"
        )

    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    return ops.lion_step(p.contiguous(), grad.contiguous(), m.contiguous(), float(lr), float(beta1), float(weight_decay))
