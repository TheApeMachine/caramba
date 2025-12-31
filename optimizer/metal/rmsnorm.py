"""Fused RMSNorm wrapper for the Metal extension."""

from __future__ import annotations

from caramba.console import logger
from typing import TYPE_CHECKING

import torch

from caramba.optimizer.runtime import METAL_SUPPORTED

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


_LOGGED = False


def metal_rmsnorm_available() -> bool:
    """Whether the runtime is capable of using the Metal RMSNorm path."""
    return bool(METAL_SUPPORTED and torch.backends.mps.is_available())


def rmsnorm_fp16(
    *,
    x: "Tensor",
    weight: "Tensor | None",
    eps: float = 1e-6,
    verbose_build: bool = False,
) -> "Tensor":
    """Fused RMSNorm (MPS/Metal) for fp16 tensors.

    Args:
        x: (..., D) fp16 tensor on MPS (contiguous required)
        weight: (D,) fp16 tensor on MPS, or None for no affine scale
        eps: epsilon for numerical stability
    """
    if x.device.type != "mps":
        raise RuntimeError("Metal RMSNorm requires device.type == 'mps'")
    if x.dtype != torch.float16:
        raise RuntimeError("Metal RMSNorm currently supports fp16 only")

    x2 = x.contiguous()
    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    global _LOGGED
    if not _LOGGED:
        logger.success("Using custom Metal kernel: RMSNorm (fp16)")
        _LOGGED = True

    if weight is None:
        return ops.rmsnorm_noweight(x2, float(eps))

    w2 = weight.to(device=x.device).to(torch.float16).contiguous()
    return ops.rmsnorm(x2, w2, float(eps))

