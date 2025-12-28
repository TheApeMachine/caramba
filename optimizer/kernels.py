"""Hardware abstraction layer (HAL) for fused kernels.

The purpose of this module is to provide a single, stable API for high-impact
ops (norms, positional encodings, attention decode, optimizer steps) while
dispatching to the best available backend:

- CUDA: Triton (when available)
- MPS: custom Metal extension
- CPU: PyTorch eager / torch.compile

This module is intentionally conservative: every op must have a correct fallback.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def rmsnorm(*, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""
    if x.device.type == "mps" and x.dtype == torch.float16:
        try:
            from optimizer.metal import metal_rmsnorm_available, rmsnorm_fp16

            if metal_rmsnorm_available():
                return rmsnorm_fp16(x=x, weight=weight, eps=float(eps))
        except Exception:
            pass

    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = (x_f * inv_rms).to(dtype=x.dtype)
    if weight is not None:
        y = y * weight
    return y


def rope_apply(*, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> Tensor:
    """Apply RoPE using cos/sin tables for the sequence window.

    Expects:
    - x: (B, H, T, D)
    - cos/sin: (T, rot_dim/2)
    """
    if x.device.type == "mps" and x.dtype == torch.float16:
        try:
            from optimizer.metal import metal_rope_available, rope_fp16

            if metal_rope_available():
                return rope_fp16(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))
        except Exception:
            pass

    T = int(x.shape[2])
    cos2 = cos[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin2 = sin[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    rot = int(rot_dim)
    x_rot = x[..., :rot]
    x_pass = x[..., rot:]
    x1 = x_rot[..., : rot // 2]
    x2 = x_rot[..., rot // 2 : rot]
    y1 = x1 * cos2 - x2 * sin2
    y2 = x1 * sin2 + x2 * cos2
    return torch.cat([y1, y2, x_pass], dim=-1)


def attention_decode(*args: Any, **kwargs: Any) -> Tensor:
    """Placeholder for fused decode attention.

    CUDA path should dispatch to Triton; MPS path should dispatch to Metal.
    """
    raise NotImplementedError("attention_decode not implemented in HAL yet")


def scan(*args: Any, **kwargs: Any) -> Tensor:
    """Placeholder for fused scan/SSM kernels."""
    raise NotImplementedError("scan not implemented in HAL yet")


def adamw_step(*args: Any, **kwargs: Any) -> None:
    """Placeholder for fused AdamW update."""
    raise NotImplementedError("adamw_step not implemented in HAL yet")


def lion_step(*args: Any, **kwargs: Any) -> None:
    """Placeholder for fused Lion update."""
    raise NotImplementedError("lion_step not implemented in HAL yet")

