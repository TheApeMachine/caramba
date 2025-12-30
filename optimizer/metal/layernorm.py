"""Fused LayerNorm wrapper for the Metal extension."""

from __future__ import annotations

from console import logger
from typing import TYPE_CHECKING

import torch

from optimizer.runtime import METAL_SUPPORTED

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


_LOGGED = False


def metal_layernorm_available() -> bool:
    """Whether the runtime is capable of using the Metal LayerNorm path."""
    return bool(METAL_SUPPORTED and torch.backends.mps.is_available())


def layernorm_fp16(
    *,
    x: "Tensor",
    weight: "Tensor | None",
    bias: "Tensor | None",
    eps: float = 1e-5,
    verbose_build: bool = False,
) -> "Tensor":
    """Fused LayerNorm (MPS/Metal) for fp16 tensors.

    Supports common forms:
    - (weight,bias) both provided (standard LayerNorm)
    - weight provided, bias None
    - weight None, bias None
    """
    if x.device.type != "mps":
        raise RuntimeError("Metal LayerNorm requires device.type == 'mps'")
    if x.dtype != torch.float16:
        raise RuntimeError("Metal LayerNorm currently supports fp16 only")
    if x.dim() < 1:
        raise RuntimeError("Metal LayerNorm expects x.dim() >= 1")

    x2 = x.contiguous()
    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    global _LOGGED
    if not _LOGGED:
        logger.success("Using custom Metal kernel: LayerNorm (fp16)")
        _LOGGED = True

    if weight is None and bias is None:
        return ops.layernorm_noweight(x2, float(eps))

    if weight is None and bias is not None:
        raise RuntimeError("Metal LayerNorm does not support bias without weight")

    assert weight is not None
    w2 = weight.to(device=x.device).to(torch.float16).contiguous()

    if bias is None:
        return ops.layernorm_weight(x2, w2, float(eps))

    b2 = bias.to(device=x.device).to(torch.float16).contiguous()
    return ops.layernorm(x2, w2, b2, float(eps))

