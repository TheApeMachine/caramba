"""CUDA/Triton LayerNorm (forward + backward)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from caramba.kernel.triton.layernorm_triton_kernels import (
    layernorm_bwd_x,
    layernorm_fwd,
    layernorm_gradb,
    layernorm_gradw,
)


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class _LNMeta:
    rows: int
    D: int
    block: int
    use_bf16: bool
    has_weight: bool
    has_bias: bool


class _LayerNormTriton:
    def _validate(self, *, x: Tensor, weight: Tensor | None, bias: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None, _LNMeta]:
        _require(x.device.type == "cuda", msg="LayerNorm Triton requires CUDA tensors.")
        _require(x.ndim >= 1, msg="LayerNorm expects x.ndim >= 1.")
        _require(x.dtype in (torch.float16, torch.bfloat16), msg="LayerNorm Triton supports fp16/bf16.")
        _require(x.stride(-1) == 1, msg="LayerNorm requires contiguous last dim (stride==1).")
        D = int(x.shape[-1])
        _require(D > 0, msg="LayerNorm requires D > 0.")
        rows = int(x.numel() // D)
        _require(rows * D == int(x.numel()), msg="LayerNorm requires x.numel divisible by D.")

        if weight is None and bias is not None:
            raise RuntimeError("LayerNorm Triton does not support bias without weight")

        w2 = None
        b2 = None
        if weight is not None:
            _require(weight.device == x.device, msg="LayerNorm weight must be on same device as x.")
            _require(weight.dtype == x.dtype, msg="LayerNorm weight dtype must match x dtype.")
            _require(weight.ndim == 1 and int(weight.numel()) == D, msg="LayerNorm weight shape mismatch.")
            _require(weight.stride(0) == 1, msg="LayerNorm weight must be contiguous.")
            w2 = weight.contiguous()
        if bias is not None:
            _require(bias.device == x.device, msg="LayerNorm bias must be on same device as x.")
            _require(bias.dtype == x.dtype, msg="LayerNorm bias dtype must match x dtype.")
            _require(bias.ndim == 1 and int(bias.numel()) == D, msg="LayerNorm bias shape mismatch.")
            _require(bias.stride(0) == 1, msg="LayerNorm bias must be contiguous.")
            b2 = bias.contiguous()

        block = 1 << (D - 1).bit_length()
        block = max(256, min(4096, block))
        meta = _LNMeta(
            rows=rows,
            D=D,
            block=block,
            use_bf16=bool(x.dtype == torch.bfloat16),
            has_weight=bool(w2 is not None),
            has_bias=bool(b2 is not None),
        )
        x2 = x.contiguous().view(rows, D)
        return x2, w2, b2, meta

    def forward(self, *, x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> tuple[Tensor, Tensor, Tensor, _LNMeta]:
        x2, w2, b2, meta = self._validate(x=x, weight=weight, bias=bias)
        _require(layernorm_fwd is not None, msg="LayerNorm Triton forward kernel is unavailable.")
        kf: Any = layernorm_fwd

        y = torch.empty_like(x2)
        mean = torch.empty((meta.rows,), device=x.device, dtype=torch.float32)
        inv = torch.empty((meta.rows,), device=x.device, dtype=torch.float32)
        kf[(meta.rows,)](
            x2,
            w2 if w2 is not None else x2,
            b2 if b2 is not None else x2,
            y,
            mean,
            inv,
            eps=float(eps),
            D=meta.D,
            stride_xr=x2.stride(0),
            stride_yr=y.stride(0),
            HAS_WEIGHT=int(meta.has_weight),
            HAS_BIAS=int(meta.has_bias),
            USE_BF16=int(meta.use_bf16),
            BLOCK=meta.block,
            num_warps=4,
        )
        return y.view_as(x), mean, inv, meta

    def backward(
        self,
        *,
        grad_y: Tensor,
        x: Tensor,
        weight: Tensor | None,
        mean: Tensor,
        inv: Tensor,
        meta: _LNMeta,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        _require(grad_y.device == x.device and grad_y.dtype == x.dtype, msg="grad_y must match x.")
        _require(layernorm_bwd_x is not None, msg="LayerNorm Triton backward-x kernel is unavailable.")
        kb: Any = layernorm_bwd_x

        gy2 = grad_y.contiguous().view(meta.rows, meta.D)
        x2 = x.contiguous().view(meta.rows, meta.D)
        gx = torch.empty_like(x2)
        kb[(meta.rows,)](
            x2,
            weight.contiguous() if weight is not None else x2,
            mean,
            inv,
            gy2,
            gx,
            D=meta.D,
            stride_xr=x2.stride(0),
            stride_gyr=gy2.stride(0),
            stride_gxr=gx.stride(0),
            HAS_WEIGHT=int(meta.has_weight),
            USE_BF16=int(meta.use_bf16),
            BLOCK=meta.block,
            num_warps=4,
        )

        if not meta.has_weight:
            return gx.view_as(x), None, None

        _require(layernorm_gradw is not None, msg="LayerNorm Triton grad_w kernel is unavailable.")
        kgw: Any = layernorm_gradw
        gw = torch.zeros((meta.D,), device=x.device, dtype=torch.float32)
        rows_per_tile = 128
        block_col = 256
        kgw[(_cdiv(meta.D, block_col), _cdiv(meta.rows, rows_per_tile))](
            x2,
            mean,
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

        if not meta.has_bias:
            return gx.view_as(x), gw.to(dtype=x.dtype), None

        _require(layernorm_gradb is not None, msg="LayerNorm Triton grad_b kernel is unavailable.")
        kgb: Any = layernorm_gradb
        gb = torch.zeros((meta.D,), device=x.device, dtype=torch.float32)
        kgb[(_cdiv(meta.D, block_col), _cdiv(meta.rows, rows_per_tile))](
            gy2,
            gb,
            rows=meta.rows,
            D=meta.D,
            stride_gyr=gy2.stride(0),
            ROWS_PER_TILE=rows_per_tile,
            BLOCK_COL=block_col,
            num_warps=4,
        )
        return gx.view_as(x), gw.to(dtype=x.dtype), gb.to(dtype=x.dtype)


class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        weight: Tensor | None,
        bias: Tensor | None,
        eps: float,
    ) -> Tensor:
        impl = _LayerNormTriton()
        y, mean, inv, meta = impl.forward(x=x, weight=weight, bias=bias, eps=float(eps))
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(x, weight if weight is not None else x, mean, inv)
        ctx.has_weight = bool(weight is not None)  # type: ignore[attr-defined]
        ctx.has_bias = bool(bias is not None)  # type: ignore[attr-defined]
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, w_or_x, mean, inv = ctx.saved_tensors
        impl: _LayerNormTriton = getattr(ctx, "impl")
        meta: _LNMeta = getattr(ctx, "meta")
        has_w = bool(getattr(ctx, "has_weight", False))
        has_b = bool(getattr(ctx, "has_bias", False))
        w = w_or_x if has_w else None
        gx, gw, gb = impl.backward(grad_y=grad_y, x=x, weight=w, mean=mean, inv=inv, meta=meta)
        if not has_b:
            gb = None
        return (gx, gw, gb, None)


def layernorm_triton(*, x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    """LayerNorm on CUDA via Triton (fp16/bf16)."""
    y = _LayerNormFn.apply(x, weight, bias, float(eps))
    if not isinstance(y, torch.Tensor):
        raise TypeError("layernorm_triton returned a non-tensor output")
    return y

