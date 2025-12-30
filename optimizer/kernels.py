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

import os
import torch
from torch import Tensor

from console import logger


def _metal_disabled(kind: str) -> bool:
    """Best-effort runtime kill-switch for Metal kernels.

    This is used to keep *correctness* (e.g. teacher parity sanity checks) independent
    from the availability of fast-path kernels.
    """
    # Global kill-switch
    if os.getenv("CARAMBA_DISABLE_METAL_KERNELS", "").strip() == "1":
        return True
    # Per-kernel kill-switch
    env = f"CARAMBA_DISABLE_METAL_{kind.upper()}"
    return os.getenv(env, "").strip() == "1"


def rmsnorm(*, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""
    if (not _metal_disabled("rmsnorm")) and x.device.type == "mps" and x.dtype == torch.float16:
        try:
            from optimizer.metal import metal_rmsnorm_available, rmsnorm_fp16

            if metal_rmsnorm_available():
                return rmsnorm_fp16(x=x, weight=weight, eps=float(eps))
        except Exception as e:
            logger.error(f"Failed to use Metal rmsnorm, continuing: {e}")

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
    if (not _metal_disabled("rope")) and x.device.type == "mps" and x.dtype == torch.float16:
        try:
            from optimizer.metal import metal_rope_available, rope_fp16

            if metal_rope_available():
                return rope_fp16(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))
        except Exception as e:
            logger.error(f"Failed to use Metal rope, continuing: {e}")

    T = int(x.shape[2])
    cos2 = cos[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin2 = sin[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    rot = int(rot_dim)
    x_rot = x[..., :rot]
    x_pass = x[..., rot:]
    # HF Llama applies rotate_half on a half-split representation:
    # y1 = x1*cos - x2*sin
    # y2 = x1*sin + x2*cos
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

