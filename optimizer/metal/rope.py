"""Fused RoPE wrapper for the Metal extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from optimizer.runtime import METAL_SUPPORTED

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


def metal_rope_available() -> bool:
    """Whether the runtime is capable of using the Metal RoPE path."""
    return bool(METAL_SUPPORTED and torch.backends.mps.is_available())


def rope_fp16(
    *,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rot_dim: int,
    verbose_build: bool = False,
) -> Tensor:
    """Apply RoPE using the Metal extension (fp16).

    Expects the same layout as `layer/rope.py`:
    - x: (B, H, T, D)
    - cos/sin: (T, rot_dim/2)
    """
    if x.device.type != "mps":
        raise RuntimeError("Metal RoPE requires device.type == 'mps'")
    if x.dtype != torch.float16:
        raise RuntimeError("Metal RoPE currently supports fp16 only")

    x2 = x.contiguous()
    cos2 = cos.to(device=x.device).to(torch.float16).contiguous()
    sin2 = sin.to(device=x.device).to(torch.float16).contiguous()

    ops = load_caramba_metal_ops(verbose=bool(verbose_build))
    return ops.rope(x2, cos2, sin2, rot_dim)

