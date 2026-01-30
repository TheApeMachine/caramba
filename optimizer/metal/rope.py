"""Fused RoPE wrapper for the Metal extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


class _AutogradCtx(Protocol):
    saved_tensors: tuple["Tensor", ...]

    def save_for_backward(self, *tensors: "Tensor") -> None: ...


def metal_rope_available() -> bool:
    """Whether the runtime is capable of using the Metal RoPE path."""
    return metal_supported()


class _MetalRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: _AutogradCtx,
        x: "Tensor",
        cos: "Tensor",
        sin: "Tensor",
        rot_dim: int,
        verbose_build: bool,
    ) -> "Tensor":
        if x.device.type != "mps":
            raise RuntimeError("Metal RoPE requires device.type == 'mps'")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError("Metal RoPE currently supports fp16/fp32 only")

        x2 = x.contiguous()
        cos2 = cos.to(device=x.device, dtype=x.dtype).contiguous()
        sin2 = sin.to(device=x.device, dtype=x.dtype).contiguous()

        # Some callsites may pass larger (cached) tables; the Metal op requires
        # exact (T, rot_dim/2) for the current window.
        T = int(x2.size(2))
        half_rot = int(rot_dim) // 2
        cos2 = cos2[:T, :half_rot].contiguous()
        sin2 = sin2[:T, :half_rot].contiguous()
        if cos2.dim() != 2 or sin2.dim() != 2 or cos2.shape != sin2.shape or cos2.shape != (T, half_rot):
            raise RuntimeError(
                "Metal RoPE: cos/sin shape mismatch before dispatch. "
                f"x={tuple(x2.shape)} rot_dim={int(rot_dim)} "
                f"expected={(T, half_rot)} cos={tuple(cos2.shape)} sin={tuple(sin2.shape)}"
            )

        ops = load_caramba_metal_ops(verbose=bool(verbose_build))

        ctx.save_for_backward(cos2, sin2)
        ctx.rot_dim = int(rot_dim)  # type: ignore[attr-defined]
        return ops.rope(x2, cos2, sin2, int(rot_dim))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: _AutogradCtx,
        grad_out: "Tensor",
    ) -> tuple["Tensor | None", ...]:
        if grad_out is None:
            raise RuntimeError("Metal RoPE backward requires grad_out")
        if grad_out.device.type != "mps":
            raise RuntimeError("Metal RoPE backward requires grad_out on MPS")

        # Ensure grad matches saved cos/sin dtype (which matched x)
        (cos, sin) = ctx.saved_tensors
        target_dtype = cos.dtype
        if grad_out.dtype != target_dtype:
            grad_out = grad_out.to(dtype=target_dtype)
        g = grad_out.contiguous()

        rot_dim = int(getattr(ctx, "rot_dim"))
        ops = load_caramba_metal_ops(verbose=False)
        grad_x = ops.rope_backward(g, cos, sin, rot_dim)
        return (grad_x, None, None, None, None)


def rope_fp16(
    *,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rot_dim: int,
    verbose_build: bool = False,
) -> Tensor:
    """Apply RoPE using the Metal extension (fp16/fp32).

    Uses the "half split" layout from `layer/rope.py`:
    - x: (B, H, T, D)
    - cos/sin: (T, rot_dim/2)
    """
    if x.device.type != "mps":
        raise RuntimeError("Metal RoPE requires device.type == 'mps'")
    if x.dtype not in (torch.float16, torch.float32):
        raise RuntimeError("Metal RoPE currently supports fp16/fp32 only")

    if not bool(x.requires_grad):
        x2 = x.contiguous()
        cos2 = cos.to(device=x.device, dtype=x.dtype).contiguous()
        sin2 = sin.to(device=x.device, dtype=x.dtype).contiguous()

        T = int(x2.size(2))
        half_rot = int(rot_dim) // 2
        cos2 = cos2[:T, :half_rot].contiguous()
        sin2 = sin2[:T, :half_rot].contiguous()
        if cos2.dim() != 2 or sin2.dim() != 2 or cos2.shape != sin2.shape or cos2.shape != (T, half_rot):
            raise RuntimeError(
                "Metal RoPE: cos/sin shape mismatch before dispatch. "
                f"x={tuple(x2.shape)} rot_dim={int(rot_dim)} "
                f"expected={(T, half_rot)} cos={tuple(cos2.shape)} sin={tuple(sin2.shape)}"
            )

        ops = load_caramba_metal_ops(verbose=bool(verbose_build))

        return ops.rope(x2, cos2, sin2, rot_dim)

    y = _MetalRoPEFn.apply(x, cos, sin, int(rot_dim), bool(verbose_build))
    if not isinstance(y, torch.Tensor):
        raise TypeError("Metal RoPE returned a non-tensor output")
    return y
