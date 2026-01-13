"""Resonant phase update Triton wrapper.

Provides an autograd-enabled fused update+normalize step on CUDA via Triton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from caramba.kernel.runtime import triton_supported
from caramba.kernel.triton.resonant_update_triton_kernels import (
    resonant_update_bwd, resonant_update_fwd
)


def _require(cond: bool, *, msg: str) -> None:
    """Requirement helper.

    Centralizes actionable error messages for kernel boundary validation.
    """
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class ResonantUpdateMeta:
    """Metadata for resonant update kernels.

    Keeps shape parameters needed for backward and validation.
    """

    D: int
    H: int
    n: int


class ResonantPhaseUpdateTritonFn(torch.autograd.Function):
    """Triton autograd wrapper for resonant phase update."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        y: Tensor,
        vr: Tensor,
        vi: Tensor,
        diag: Tensor,
        scale: float,
        damping: float,
        zero_diag: bool,
    ) -> tuple[Tensor, Tensor]:
        if not triton_supported():
            raise RuntimeError(
                "ResonantPhaseUpdateTriton requires Triton but it is not installed.\n"
                "Fix: install triton and ensure CUDA is available."
            )
        _require(x.device.type == "cuda", msg="ResonantPhaseUpdateTriton requires CUDA tensors.")
        _require(x.dtype == torch.float32, msg="ResonantPhaseUpdateTriton requires x float32.")
        _require(y.dtype == torch.float32 and vr.dtype == torch.float32 and vi.dtype == torch.float32, msg="ResonantPhaseUpdateTriton requires fp32 tensors.")
        _require(x.shape == y.shape == vr.shape == vi.shape, msg="ResonantPhaseUpdateTriton requires matching x/y/vr/vi shapes.")
        _require(x.ndim == 3, msg="ResonantPhaseUpdateTriton expects (BT,H,D) tensors.")
        _require(diag.ndim == 2, msg="ResonantPhaseUpdateTriton expects diag (H,D).")
        BT, H, D = x.shape
        _require(tuple(diag.shape) == (int(H), int(D)), msg="diag must have shape (H,D) matching x.")

        n = int(BT) * int(H) * int(D)
        meta = ResonantUpdateMeta(D=int(D), H=int(H), n=int(n))

        x2 = x.contiguous()
        y2 = y.contiguous()
        vr2 = vr.contiguous()
        vi2 = vi.contiguous()
        diag2 = diag.contiguous()

        xo = torch.empty_like(x2)
        yo = torch.empty_like(y2)
        a = torch.empty_like(x2)
        b = torch.empty_like(y2)
        inv_r = torch.empty_like(x2)

        import triton  # type: ignore[reportMissingImports]

        kf: Any = resonant_update_fwd
        _require(kf is not None, msg="ResonantPhaseUpdateTriton forward kernel is unavailable.")
        grid = (triton.cdiv(meta.n, 256),)
        kf[grid](
            x2,
            y2,
            vr2,
            vi2,
            diag2,
            xo,
            yo,
            a,
            b,
            inv_r,
            n_elements=meta.n,
            D=meta.D,
            H=meta.H,
            inv_D=float(1.0 / float(meta.D)),
            scale=float(scale),
            damping=float(damping),
            zero_diag=1 if bool(zero_diag) else 0,
            BLOCK=256,
            num_warps=4,
        )

        ctx.save_for_backward(x2, y2, diag2, a, b, inv_r)
        ctx.meta = meta
        ctx.scale = float(scale)
        ctx.damping = float(damping)
        ctx.zero_diag = bool(zero_diag)
        return xo, yo

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_xo: Tensor,
        grad_yo: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, y, diag, a, b, inv_r = ctx.saved_tensors
        meta: ResonantUpdateMeta = ctx.meta
        scale = float(ctx.scale)
        damping = float(ctx.damping)
        zero_diag = bool(ctx.zero_diag)

        _require(grad_xo.device.type == "cuda", msg="grad_xo must be CUDA for ResonantPhaseUpdateTriton.")
        gx = torch.empty_like(x)
        gy = torch.empty_like(y)
        gvr = torch.empty_like(x)
        gvi = torch.empty_like(x)

        import triton  # type: ignore[reportMissingImports]

        kb: Any = resonant_update_bwd
        _require(kb is not None, msg="ResonantPhaseUpdateTriton backward kernel is unavailable.")
        grid = (triton.cdiv(meta.n, 256),)
        kb[grid](
            grad_xo.contiguous(),
            grad_yo.contiguous(),
            diag,
            gvr,
            gvi,
            gx,
            gy,
            a,
            b,
            inv_r,
            n_elements=meta.n,
            D=meta.D,
            H=meta.H,
            inv_D=float(1.0 / float(meta.D)),
            scale=float(scale),
            damping=float(damping),
            zero_diag=1 if bool(zero_diag) else 0,
            BLOCK=256,
            num_warps=4,
        )

        # No gradient for diag (derived from patterns).
        return gx, gy, gvr, gvi, None, None, None, None


class ResonantPhaseUpdateTriton:
    """CUDA resonant phase update.

    Wraps the fused Triton kernel behind a small object so it can be composed
    by higher-level routing logic.
    """

    def forward(self, *, x: Tensor, y: Tensor, vr: Tensor, vi: Tensor, diag: Tensor, scale: float, damping: float, zero_diag: bool) -> tuple[Tensor, Tensor]:
        out = ResonantPhaseUpdateTritonFn.apply(x, y, vr, vi, diag, float(scale), float(damping), bool(zero_diag))
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], torch.Tensor) and isinstance(out[1], torch.Tensor)):
            raise TypeError("ResonantPhaseUpdateTriton returned invalid outputs")
        return out[0], out[1]

