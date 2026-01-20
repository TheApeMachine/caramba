"""CUDA/Triton RMSNorm.

Forward saves per-row `inv_rms` for an efficient custom backward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from optimizer.rmsnorm_triton_kernels import rmsnorm_bwd_w, rmsnorm_bwd_x, rmsnorm_bwd_x_noweight, rmsnorm_fwd


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class _RMSMeta:
    rows: int
    D: int
    block: int


class _RMSNormTriton:
    """Fused RMSNorm on CUDA via Triton."""

    def _validate(self, *, x: Tensor, weight: Tensor | None) -> tuple[Tensor, Tensor | None, _RMSMeta]:
        _require(x.device.type == "cuda", msg="RMSNorm Triton requires CUDA tensors.")
        _require(x.ndim >= 1, msg="RMSNorm expects x.ndim >= 1.")
        _require(x.dtype in (torch.float16, torch.bfloat16), msg="RMSNorm Triton supports fp16/bf16.")
        _require(x.stride(-1) == 1, msg="RMSNorm requires contiguous last dim (stride==1).")
        if weight is not None:
            _require(weight.device == x.device, msg="RMSNorm weight must be on the same device as x.")
            _require(weight.dtype == x.dtype, msg="RMSNorm weight dtype must match x dtype.")
            _require(weight.ndim == 1, msg="RMSNorm weight must be 1D.")
            _require(weight.stride(0) == 1, msg="RMSNorm weight must be contiguous.")

        D = int(x.shape[-1])
        _require(D > 0, msg="RMSNorm requires D > 0.")
        if weight is not None:
            _require(int(weight.numel()) == D, msg="RMSNorm weight numel must match x.shape[-1].")
        rows = int(x.numel() // D)
        _require(rows * D == int(x.numel()), msg="RMSNorm requires x.numel divisible by D.")

        block = 1 << (D - 1).bit_length()
        block = max(256, min(4096, block))
        meta = _RMSMeta(rows=rows, D=D, block=block)
        x2 = x.contiguous().view(rows, D)
        w2 = weight.contiguous() if weight is not None else None
        return x2, w2, meta

    def forward(self, *, x: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor, _RMSMeta]:
        x2, w2, meta = self._validate(x=x, weight=weight)
        _require(rmsnorm_fwd is not None, msg="RMSNorm Triton forward kernel is unavailable.")
        kf: Any = rmsnorm_fwd

        y = torch.empty_like(x2, dtype=torch.float32)
        inv = torch.empty((meta.rows,), device=x.device, dtype=torch.float32)
        has_w = int(w2 is not None)
        kf[(meta.rows,)](
            x2,
            w2 if w2 is not None else x2,  # unused when has_w==0
            y,
            inv,
            eps=float(eps),
            D=meta.D,
            stride_xr=x2.stride(0),
            stride_yr=y.stride(0),
            HAS_WEIGHT=has_w,
            BLOCK=meta.block,
            num_warps=4,
        )
        return y.to(dtype=x.dtype).view_as(x), inv, meta

    def backward(
        self,
        *,
        grad_y: Tensor,
        x: Tensor,
        weight: Tensor | None,
        inv: Tensor,
        meta: _RMSMeta,
    ) -> tuple[Tensor, Tensor | None]:
        _require(grad_y.device == x.device, msg="grad_y must be on same device as x.")
        _require(grad_y.dtype == x.dtype, msg="grad_y dtype must match x dtype.")
        gy2 = grad_y.contiguous().view(meta.rows, meta.D)
        x2 = x.contiguous().view(meta.rows, meta.D)

        gx = torch.empty((meta.rows, meta.D), device=x.device, dtype=torch.float32)

        if weight is None:
            _require(rmsnorm_bwd_x_noweight is not None, msg="RMSNorm Triton backward-x (no weight) kernel is unavailable.")
            kb: Any = rmsnorm_bwd_x_noweight
            kb[(meta.rows,)](
                x2,
                inv,
                gy2,
                gx,
                D=meta.D,
                stride_xr=x2.stride(0),
                stride_gyr=gy2.stride(0),
                stride_gxr=gx.stride(0),
                BLOCK=meta.block,
                num_warps=4,
            )
            return gx.to(dtype=x.dtype).view_as(x), None

        _require(rmsnorm_bwd_x is not None and rmsnorm_bwd_w is not None, msg="RMSNorm Triton backward kernels are unavailable.")
        kx: Any = rmsnorm_bwd_x
        kw: Any = rmsnorm_bwd_w
        w2 = weight.contiguous()

        kx[(meta.rows,)](
            x2,
            w2,
            inv,
            gy2,
            gx,
            D=meta.D,
            stride_xr=x2.stride(0),
            stride_gyr=gy2.stride(0),
            stride_gxr=gx.stride(0),
            BLOCK=meta.block,
            num_warps=4,
        )

        # Kernel uses atomic adds across row tiles; initialize to zero.
        gw = torch.zeros((meta.D,), device=x.device, dtype=torch.float32)
        block_col = 256
        rows_per_tile = 8
        kw[(_cdiv(meta.D, block_col), _cdiv(meta.rows, rows_per_tile))](
            x2,
            inv,
            gy2,
            gw,
            rows=meta.rows,
            D=meta.D,
            stride_xr=x2.stride(0),
            stride_gyr=gy2.stride(0),
            ROWS_PER_TILE=rows_per_tile,
            BLOCK_COL=block_col,
            num_warps=4,
        )
        return gx.to(dtype=x.dtype).view_as(x), gw.to(dtype=weight.dtype)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        weight: Tensor | None,
        eps: float,
    ) -> Tensor:
        impl = _RMSNormTriton()
        y, inv, meta = impl.forward(x=x, weight=weight, eps=float(eps))
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(x, weight if weight is not None else x, inv)
        ctx.has_weight = bool(weight is not None)  # type: ignore[attr-defined]
        ctx.eps = float(eps)  # type: ignore[attr-defined]
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, w_or_x, inv = ctx.saved_tensors
        has_w = bool(getattr(ctx, "has_weight", False))
        weight = w_or_x if has_w else None
        impl: _RMSNormTriton = getattr(ctx, "impl")
        meta: _RMSMeta = getattr(ctx, "meta")
        gx, gw = impl.backward(grad_y=grad_y, x=x, weight=weight, inv=inv, meta=meta)
        return (gx, gw, None)


def rmsnorm_triton(*, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm on CUDA via Triton (fp16/bf16)."""
    y = _RMSNormFn.apply(x, weight, float(eps))
    if not isinstance(y, torch.Tensor):
        raise TypeError("rmsnorm_triton returned a non-tensor output")
    return y
