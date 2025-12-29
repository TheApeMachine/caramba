"""Fused Lion optimizer update wrapper for the Metal extension."""

from __future__ import annotations

from console import logger
from typing import TYPE_CHECKING

import torch

from optimizer.runtime import METAL_SUPPORTED

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


_LOGGED = False


def metal_lion_available() -> bool:
    return bool(METAL_SUPPORTED and torch.backends.mps.is_available())


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
    if p.dtype != torch.float16 or grad.dtype != torch.float16 or m.dtype != torch.float16:
        raise RuntimeError("Metal Lion currently supports fp16 tensors only")
    if p.shape != grad.shape or p.shape != m.shape:
        raise RuntimeError(
            f"Metal Lion requires matching shapes for p/grad/m, got p={tuple(p.shape)}, grad={tuple(grad.shape)}, m={tuple(m.shape)}"
        )

    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    global _LOGGED
    if not _LOGGED:
        logger.success("Using custom Metal kernel: Lion optimizer (fp16)")
        _LOGGED = True

    return ops.lion_step(p.contiguous(), grad.contiguous(), m.contiguous(), float(lr), float(beta1), float(weight_decay))

