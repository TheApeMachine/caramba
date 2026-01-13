"""CUDA/Triton RoPE (half-split layout) with backward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from caramba.kernel.triton.rope_triton_kernels import rope_bwd, rope_fwd


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class _RoPEMeta:
    B: int
    H: int
    T: int
    D: int
    rot: int
    half: int
    block: int
    use_bf16: bool


class _RoPETriton:
    def _validate(self, *, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> _RoPEMeta:
        _require(x.device.type == "cuda", msg="RoPE Triton requires CUDA tensors.")
        _require(x.ndim == 4, msg="RoPE expects x shape (B,H,T,D).")
        _require(x.dtype in (torch.float16, torch.bfloat16), msg="RoPE Triton supports fp16/bf16.")
        _require(x.is_contiguous(), msg="RoPE Triton requires x contiguous.")
        _require(cos.device == x.device and sin.device == x.device, msg="cos/sin must be on same device as x.")
        _require(cos.dtype == x.dtype and sin.dtype == x.dtype, msg="cos/sin dtype must match x dtype.")
        _require(cos.is_contiguous() and sin.is_contiguous(), msg="cos/sin must be contiguous.")
        _require(cos.ndim == 2 and sin.ndim == 2, msg="cos/sin must be (T, rot/2).")

        B, H, T, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
        rot = int(rot_dim)
        _require(rot > 0 and (rot % 2) == 0, msg="rot_dim must be positive and even.")
        _require(rot <= D, msg="rot_dim must be <= head_dim.")
        half = rot // 2
        _require(int(cos.shape[0]) == T and int(cos.shape[1]) == half, msg="cos shape mismatch.")
        _require(int(sin.shape[0]) == T and int(sin.shape[1]) == half, msg="sin shape mismatch.")

        block = 256
        return _RoPEMeta(B=B, H=H, T=T, D=D, rot=rot, half=half, block=block, use_bf16=bool(x.dtype == torch.bfloat16))

    def forward(self, *, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> tuple[Tensor, _RoPEMeta]:
        meta = self._validate(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))
        _require(rope_fwd is not None, msg="RoPE Triton forward kernel is unavailable.")
        kf: Any = rope_fwd

        n_vec = meta.B * meta.H * meta.T
        y = torch.empty_like(x).view(n_vec, meta.D)
        x2 = x.view(n_vec, meta.D)
        grid = (n_vec, _cdiv(meta.D, meta.block))
        kf[grid](
            x2,
            cos,
            sin,
            y,
            T=meta.T,
            D=meta.D,
            ROT=meta.rot,
            HALF=meta.half,
            stride_xv=x2.stride(0),
            stride_xt=x2.stride(1),
            stride_yv=y.stride(0),
            stride_yt=y.stride(1),
            stride_cos_t=cos.stride(0),
            stride_cos_h=cos.stride(1),
            stride_sin_t=sin.stride(0),
            stride_sin_h=sin.stride(1),
            USE_BF16=int(meta.use_bf16),
            BLOCK=meta.block,
            num_warps=4,
        )
        return y.view_as(x), meta

    def backward(self, *, grad_y: Tensor, cos: Tensor, sin: Tensor, meta: _RoPEMeta) -> Tensor:
        _require(rope_bwd is not None, msg="RoPE Triton backward kernel is unavailable.")
        kb: Any = rope_bwd
        _require(grad_y.is_contiguous(), msg="RoPE backward requires grad_y contiguous.")

        n_vec = meta.B * meta.H * meta.T
        gy2 = grad_y.view(n_vec, meta.D)
        gx = torch.empty_like(gy2)
        grid = (n_vec, _cdiv(meta.D, meta.block))
        kb[grid](
            gy2,
            cos,
            sin,
            gx,
            T=meta.T,
            D=meta.D,
            ROT=meta.rot,
            HALF=meta.half,
            stride_gyv=gy2.stride(0),
            stride_gyt=gy2.stride(1),
            stride_gxv=gx.stride(0),
            stride_gxt=gx.stride(1),
            stride_cos_t=cos.stride(0),
            stride_cos_h=cos.stride(1),
            stride_sin_t=sin.stride(0),
            stride_sin_h=sin.stride(1),
            USE_BF16=int(meta.use_bf16),
            BLOCK=meta.block,
            num_warps=4,
        )
        return gx.view_as(grad_y)


class _RoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        rot_dim: int,
    ) -> Tensor:
        impl = _RoPETriton()
        y, meta = impl.forward(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(cos, sin)
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        (cos, sin) = ctx.saved_tensors
        impl: _RoPETriton = getattr(ctx, "impl")
        meta: _RoPEMeta = getattr(ctx, "meta")
        gx = impl.backward(grad_y=grad_y.contiguous(), cos=cos, sin=sin, meta=meta)
        return (gx, None, None, None)


def rope_triton(*, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> Tensor:
    """RoPE on CUDA via Triton (fp16/bf16)."""
    y = _RoPEFn.apply(x, cos, sin, int(rot_dim))
    if not isinstance(y, torch.Tensor):
        raise TypeError("rope_triton returned a non-tensor output")
    return y

