"""Fused LayerNorm wrapper for the Metal extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from caramba.optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


class _AutogradCtx(Protocol):
    saved_tensors: tuple["Tensor", ...]

    def save_for_backward(self, *tensors: "Tensor") -> None: ...


def metal_layernorm_available() -> bool:
    """Whether the runtime is capable of using the Metal LayerNorm path."""
    return metal_supported()


class _MetalLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: _AutogradCtx,
        x: "Tensor",
        weight: "Tensor | None",
        bias: "Tensor | None",
        eps: float,
        verbose_build: bool,
    ) -> "Tensor":
        if x.device.type != "mps":
            raise RuntimeError("Metal LayerNorm requires device.type == 'mps'")
        if x.dtype != torch.float16:
            raise RuntimeError("Metal LayerNorm currently supports fp16 only")
        if x.dim() < 1:
            raise RuntimeError("Metal LayerNorm expects x.dim() >= 1")

        d_model = int(x.shape[-1])
        if weight is not None and int(weight.shape[0]) != d_model:
            raise ValueError(
                f"layernorm weight size mismatch: weight.shape[0]={int(weight.shape[0])} "
                f"but x.shape[-1]={d_model}"
            )
        if bias is not None and int(bias.shape[0]) != d_model:
            raise ValueError(
                f"layernorm bias size mismatch: bias.shape[0]={int(bias.shape[0])} "
                f"but x.shape[-1]={d_model}"
            )
        if weight is None and bias is not None:
            raise RuntimeError("Metal LayerNorm does not support bias without weight")

        x2 = x.contiguous()
        ops = load_caramba_metal_ops(verbose=bool(verbose_build))

        has_weight = weight is not None
        has_bias = bias is not None
        ctx.has_weight = bool(has_weight)  # type: ignore[attr-defined]
        ctx.has_bias = bool(has_bias)  # type: ignore[attr-defined]

        if not has_weight:
            out, mean, inv = ops.layernorm_noweight_forward_with_stats(x2, float(eps))
            ctx.save_for_backward(x2, mean, inv)
            return out

        w2 = weight.to(device=x.device, dtype=torch.float16).contiguous()
        if not has_bias:
            out, mean, inv = ops.layernorm_weight_forward_with_stats(x2, w2, float(eps))
            ctx.save_for_backward(x2, w2, mean, inv)
            return out

        b2 = bias.to(device=x.device, dtype=torch.float16).contiguous()
        out, mean, inv = ops.layernorm_forward_with_stats(x2, w2, b2, float(eps))
        ctx.save_for_backward(x2, w2, mean, inv)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: _AutogradCtx,
        grad_out: "Tensor",
    ) -> tuple["Tensor | None", ...]:
        if grad_out is None:
            raise RuntimeError("Metal LayerNorm backward requires grad_out")
        if grad_out.device.type != "mps":
            raise RuntimeError("Metal LayerNorm backward requires grad_out on MPS")
        if grad_out.dtype != torch.float16:
            grad_out = grad_out.to(dtype=torch.float16)
        g = grad_out.contiguous()

        ops = load_caramba_metal_ops(verbose=False)
        has_weight = bool(getattr(ctx, "has_weight", False))
        has_bias = bool(getattr(ctx, "has_bias", False))

        saved = ctx.saved_tensors
        if not has_weight:
            if len(saved) != 3:
                raise RuntimeError("Metal LayerNorm backward: invalid saved tensor state (noweight)")
            x, mean, inv = saved
            grad_x = ops.layernorm_backward_x_noweight(g, x, mean, inv)
            return (grad_x, None, None, None, None)

        if len(saved) != 4:
            raise RuntimeError("Metal LayerNorm backward: invalid saved tensor state (weight)")
        x, w, mean, inv = saved
        grad_x = ops.layernorm_backward_x(g, x, w, mean, inv)
        grad_w = ops.layernorm_backward_w(g, x, mean, inv)
        grad_b = ops.layernorm_backward_b(g) if has_bias else None
        return (grad_x, grad_w, grad_b, None, None)


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

    d_model = int(x.shape[-1])
    if weight is not None and int(weight.shape[0]) != d_model:
        raise ValueError(
            f"layernorm weight size mismatch: weight.shape[0]={int(weight.shape[0])} "
            f"but x.shape[-1]={d_model}"
        )
    if bias is not None and int(bias.shape[0]) != d_model:
        raise ValueError(
            f"layernorm bias size mismatch: bias.shape[0]={int(bias.shape[0])} "
            f"but x.shape[-1]={d_model}"
        )
    if weight is None and bias is not None:
        raise RuntimeError("Metal LayerNorm does not support bias without weight")

    needs_grad = bool(x.requires_grad) or (weight is not None and bool(weight.requires_grad)) or (bias is not None and bool(bias.requires_grad))
    if needs_grad:
        y = _MetalLayerNormFn.apply(x, weight, bias, float(eps), bool(verbose_build))
        if not isinstance(y, torch.Tensor):
            raise TypeError("Metal LayerNorm returned a non-tensor output")
        return y

    x2 = x.contiguous()
    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    if weight is None:
        return ops.layernorm_noweight(x2, float(eps))

    w2 = weight.to(device=x.device, dtype=torch.float16).contiguous()
    if bias is None:
        return ops.layernorm_weight(x2, w2, float(eps))

    b2 = bias.to(device=x.device, dtype=torch.float16).contiguous()
    return ops.layernorm(x2, w2, b2, float(eps))
