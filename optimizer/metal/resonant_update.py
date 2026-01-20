"""Metal resonant phase update wrapper.

Provides a fused update+normalize step on MPS via a Metal extension.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor

from optimizer.metal.resonant_jit import load_caramba_metal_resonant_ops


class AutogradContext(Protocol):
    """Minimal autograd ctx protocol."""

    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class MetalResonantPhaseUpdateFn(torch.autograd.Function):
    """Autograd wrapper for Metal resonant update."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: AutogradContext,
        x: Tensor,
        y: Tensor,
        vr: Tensor,
        vi: Tensor,
        diag: Tensor,
        scale: float,
        damping: float,
        zero_diag: bool,
    ) -> tuple[Tensor, Tensor]:
        if x.device.type != "mps":
            raise RuntimeError("Metal resonant update requires MPS tensors")
        if x.dtype != torch.float32:
            raise RuntimeError("Metal resonant update requires fp32 tensors")
        if x.shape != y.shape or x.shape != vr.shape or x.shape != vi.shape:
            raise RuntimeError("Metal resonant update requires matching x/y/vr/vi shapes")
        if diag.ndim != 2:
            raise RuntimeError("Metal resonant update requires diag (H,D)")

        ops = load_caramba_metal_resonant_ops(verbose=False)
        packed = ops.resonant_update_forward_fp32(
            x.contiguous(),
            y.contiguous(),
            vr.contiguous(),
            vi.contiguous(),
            diag.contiguous(),
            float(scale),
            float(damping),
            bool(zero_diag),
        )
        if not isinstance(packed, torch.Tensor) or packed.ndim != 4 or int(packed.size(0)) != 5:
            raise TypeError("Metal resonant update returned invalid packed tensor")
        xo, yo, a, b, inv_r = packed[0], packed[1], packed[2], packed[3], packed[4]
        ctx.save_for_backward(x, y, diag, a, b, inv_r)
        ctx.scale = float(scale)  # type: ignore[attr-defined]
        ctx.damping = float(damping)  # type: ignore[attr-defined]
        ctx.zero_diag = bool(zero_diag)  # type: ignore[attr-defined]
        return xo, yo

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_xo: Tensor,
        grad_yo: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, y, diag, a, b, inv_r = ctx.saved_tensors
        ops = load_caramba_metal_resonant_ops(verbose=False)
        scale = float(ctx.scale)
        damping = float(ctx.damping)
        zero_diag = bool(ctx.zero_diag)

        gx, gy, gvr, gvi = ops.resonant_update_backward_fp32(
            grad_xo.contiguous(),
            grad_yo.contiguous(),
            x.contiguous(),
            y.contiguous(),
            diag.contiguous(),
            a.contiguous(),
            b.contiguous(),
            inv_r.contiguous(),
            float(scale),
            float(damping),
            bool(zero_diag),
        )
        return gx, gy, gvr, gvi, None, None, None, None


class MetalResonantPhaseUpdate:
    """Metal resonant phase update.

    Used by resonant routing logic to update phase state efficiently on MPS.
    """

    def forward(self, *, x: Tensor, y: Tensor, vr: Tensor, vi: Tensor, diag: Tensor, scale: float, damping: float, zero_diag: bool) -> tuple[Tensor, Tensor]:
        out = MetalResonantPhaseUpdateFn.apply(x, y, vr, vi, diag, float(scale), float(damping), bool(zero_diag))
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], Tensor) and isinstance(out[1], Tensor)):
            raise TypeError("MetalResonantPhaseUpdate returned invalid outputs")
        return out[0], out[1]

