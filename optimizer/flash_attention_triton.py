"""CUDA/Triton FlashAttention forward+backward.

This provides a deterministic, custom CUDA attention training kernel:
- Full-sequence attention (no KV-cache; q/k/v have same T)
- Causal (optional)
- No dropout and no arbitrary masks (by design; enforce with clear errors)

This is intended to replace PyTorch SDPA for CUDA training in the standard and
decoupled attention layers when conditions match the kernel contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from console import logger
from optimizer.runtime import triton_supported


# Keep this module importable on non-CUDA installs (e.g., MPS-only laptops).
# The *symbols* exist, but kernels stay as None unless Triton is present.
flash_attn_fwd: object | None = None
flash_attn_bwd_preprocess: object | None = None
flash_attn_bwd_dkv: object | None = None
flash_attn_bwd_dq: object | None = None

if not TYPE_CHECKING and triton_supported():
    from optimizer.flash_attention_triton_kernels_bwd import (
        flash_attn_bwd_dkv,
        flash_attn_bwd_dq,
        flash_attn_bwd_preprocess,
    )
    from optimizer.flash_attention_triton_kernels_fwd import flash_attn_fwd

_TRITON_DEBUG: bool = False


@dataclass(frozen=True, slots=True)
class _FlashMeta:
    Z: int
    T: int
    D: int
    causal: bool
    scale: float
    dropout_p: float
    seed: int
    block_m: int
    block_n: int
    block_d: int


class _FlashAttentionTriton:
    """FlashAttention on CUDA via Triton kernels."""

    def _require(self, cond: bool, *, msg: str) -> None:
        if not cond:
            raise RuntimeError(msg)

    def _cdiv(self, n: int, d: int) -> int:
        return (int(n) + int(d) - 1) // int(d)

    def _validate_inputs(self, *, q: Tensor, k: Tensor, v: Tensor) -> tuple[int, int, int, int]:
        self._require(q.device.type == "cuda", msg="FlashAttention Triton requires CUDA tensors.")
        self._require(k.device == q.device and v.device == q.device, msg="q/k/v must be on the same CUDA device.")
        self._require(q.ndim == 4 and k.ndim == 4 and v.ndim == 4, msg="FlashAttention expects q/k/v shape (B,H,T,D).")
        self._require(q.shape == k.shape == v.shape, msg="FlashAttention requires q/k/v to have identical shapes (B,H,T,D).")
        self._require(q.dtype == k.dtype == v.dtype, msg="FlashAttention requires q/k/v dtypes to match.")
        self._require(q.dtype in (torch.float16, torch.bfloat16, torch.float32), msg="FlashAttention supports fp16/bf16/fp32.")
        self._require(q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), msg="FlashAttention requires contiguous q/k/v.")
        self._require(q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, msg="FlashAttention requires contiguous last dim.")
        B, H, T, D = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
        self._require(T > 0 and D > 0, msg="FlashAttention requires non-empty T and D.")
        self._require(D <= 128, msg=f"FlashAttention Triton currently supports head_dim <= 128, got head_dim={D}.")
        return B, H, T, D

    def _flatten_bh(self, *, x: Tensor, Z: int, T: int, D: int) -> Tensor:
        if _TRITON_DEBUG:
            logger.info(f"Flattening tensor with shape: {x.shape} to (Z, T, D): {Z}, {T}, {D}")
        return x.reshape(Z, T, D)

    def _meta(self, *, Z: int, T: int, D: int, causal: bool, scale: float, dropout_p: float, seed: int) -> _FlashMeta:
        if _TRITON_DEBUG:
            logger.info(
                f"Creating FlashAttention meta with Z: {Z}, T: {T}, D: {D}, causal: {causal}, scale: {scale}, dropout_p: {dropout_p}, seed: {seed}"
            )
        return _FlashMeta(
            Z=Z,
            T=T,
            D=D,
            causal=bool(causal),
            scale=float(scale),
            dropout_p=float(dropout_p),
            seed=int(seed),
            block_m=64,
            block_n=64,
            block_d=32 if int(D) <= 32 else (64 if int(D) <= 64 else 128),
        )

    def _launch_fwd(self, *, q: Tensor, k: Tensor, v: Tensor, meta: _FlashMeta) -> tuple[Tensor, Tensor]:
        if _TRITON_DEBUG:
            logger.info(f"[TRITON] flash fwd launch q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} meta={meta}")
        self._require(flash_attn_fwd is not None, msg="FlashAttention Triton forward kernel is unavailable.")
        kf: Any = flash_attn_fwd
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((meta.Z, meta.T), device=q.device, dtype=torch.float32)
        grid = (meta.Z, self._cdiv(meta.T, meta.block_m))
        num_n = self._cdiv(meta.T, meta.block_n)
        # Important: Triton expects tl.constexpr values to be plain Python ints.
        # Under torch.compile / dynamo these can become SymInt-like; force them.
        sqz, sqt, sqd = (int(q.stride(0)), int(q.stride(1)), int(q.stride(2)))
        skz, skt, skd = (int(k.stride(0)), int(k.stride(1)), int(k.stride(2)))
        svz, svt, svd = (int(v.stride(0)), int(v.stride(1)), int(v.stride(2)))
        soz, sot, sod = (int(out.stride(0)), int(out.stride(1)), int(out.stride(2)))
        slsez, slset = (int(lse.stride(0)), int(lse.stride(1)))

        if _TRITON_DEBUG:
            logger.info(
                "[TRITON] flash fwd strides "
                f"q=({sqz},{sqt},{sqd}) k=({skz},{skt},{skd}) v=({svz},{svt},{svd}) "
                f"o=({soz},{sot},{sod}) lse=({slsez},{slset}) grid={grid} NUM_N={num_n} num_warps=4"
            )
            mod = sys.modules.get(getattr(kf, "__module__", ""))
            logger.info(f"[TRITON] flash_attn_fwd module={getattr(mod, '__file__', None)}")
            logger.info(f"[TRITON] flash_attn_fwd arg_names={getattr(kf, 'arg_names', None)}")
            logger.info(f"[TRITON] flash_attn_fwd constexprs={getattr(kf, 'constexprs', None)}")
            logger.info(f"[TRITON] meta={meta.T}, {meta.D}, {meta.block_m}, {meta.block_n}, {meta.block_d}")

        # Fail fast with an actionable message if the JIT signature doesn't match.
        arg_names = getattr(kf, "arg_names", None)
        if isinstance(arg_names, (list, tuple)) and "stride_qz" not in arg_names:
            mod = sys.modules.get(getattr(kf, "__module__", ""))
            raise RuntimeError(
                "FlashAttention Triton kernel signature mismatch: 'stride_qz' is not an argument of flash_attn_fwd.\n"
                f"flash_attn_fwd module file: {getattr(mod, '__file__', None)}\n"
                f"flash_attn_fwd arg_names: {arg_names}\n"
                "This usually means you're importing a different kernels_fwd.py than you think, or a stale build is being used."
            )
        if isinstance(arg_names, (list, tuple)) and "T" not in arg_names:
            mod = sys.modules.get(getattr(kf, "__module__", ""))
            raise RuntimeError(
                "FlashAttention Triton kernel signature mismatch: 'T' is not an argument of flash_attn_fwd.\n"
                f"flash_attn_fwd module file: {getattr(mod, '__file__', None)}\n"
                f"flash_attn_fwd arg_names: {arg_names}\n"
                "This usually means you're importing a different kernels_fwd.py than you think, or a stale build is being used."
            )
        if isinstance(arg_names, (list, tuple)) and "BLOCK_M" not in arg_names:
            mod = sys.modules.get(getattr(kf, "__module__", ""))
            raise RuntimeError(
                "FlashAttention Triton kernel signature mismatch: 'BLOCK_M' is not an argument of flash_attn_fwd.\n"
                f"flash_attn_fwd module file: {getattr(mod, '__file__', None)}\n"
                f"flash_attn_fwd arg_names: {arg_names}\n"
                "This usually means you're importing a different kernels_fwd.py than you think, or a stale build is being used."
            )
        kf[grid](
            q, k, v, out, lse, meta.seed,
            stride_qz=sqz, stride_qt=sqt, stride_qd=sqd,
            stride_kz=skz, stride_kt=skt, stride_kd=skd,
            stride_vz=svz, stride_vt=svt, stride_vd=svd,
            stride_oz=soz, stride_ot=sot, stride_od=sod,
            stride_lsez=slsez, stride_lset=slset,
            T=meta.T, D=meta.D,
            SM_SCALE=float(meta.scale), CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n, BLOCK_D=meta.block_d,
            num_warps=4,
        )
        return out, lse

    def forward(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        dropout_p: float,
        seed: int,
    ) -> tuple[Tensor, Tensor, _FlashMeta]:
        B, H, T, D = self._validate_inputs(q=q, k=k, v=v)
        Z = B * H
        meta = self._meta(Z=Z, T=T, D=D, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p), seed=int(seed))
        q2 = self._flatten_bh(x=q, Z=Z, T=T, D=D)
        k2 = self._flatten_bh(x=k, Z=Z, T=T, D=D)
        v2 = self._flatten_bh(x=v, Z=Z, T=T, D=D)
        out, lse = self._launch_fwd(q=q2, k=k2, v=v2, meta=meta)
        return out.to(dtype=q.dtype).reshape_as(q), lse, meta

    def _launch_preprocess(self, *, out: Tensor, grad_out: Tensor, meta: _FlashMeta) -> Tensor:
        self._require(flash_attn_bwd_preprocess is not None, msg="FlashAttention preprocess kernel is unavailable.")
        kp: Any = flash_attn_bwd_preprocess
        delta = torch.empty((meta.Z, meta.T), device=out.device, dtype=torch.float32)
        grid = (meta.Z, self._cdiv(meta.T, meta.block_m))
        soz, sot, sod = (int(out.stride(0)), int(out.stride(1)), int(out.stride(2)))
        sdoz, sdot, sdod = (int(grad_out.stride(0)), int(grad_out.stride(1)), int(grad_out.stride(2)))
        sdeltaz, sdeltat = (int(delta.stride(0)), int(delta.stride(1)))
        if _TRITON_DEBUG:
            logger.info(
                f"Launching FlashAttention preprocess kernel with out: {out.shape}, grad_out: {grad_out.shape}, delta: {delta.shape}"
            )
        kp[grid](
            out, grad_out, delta,
            stride_oz=soz, stride_ot=sot, stride_od=sod,
            stride_doz=sdoz, stride_dot=sdot, stride_dod=sdod,
            stride_deltaz=sdeltaz, stride_deltat=sdeltat,
            Z=meta.Z, T=meta.T, D=meta.D,
            BLOCK_M=meta.block_m, BLOCK_D=meta.block_d,
            num_warps=4,
        )
        return delta

    def _launch_dkv(self, *, q: Tensor, k: Tensor, v: Tensor, grad_out: Tensor, lse: Tensor, delta: Tensor, meta: _FlashMeta) -> tuple[Tensor, Tensor]:
        self._require(flash_attn_bwd_dkv is not None, msg="FlashAttention backward (dK/dV) kernel is unavailable.")
        kdkv: Any = flash_attn_bwd_dkv
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        grid = (meta.Z, self._cdiv(meta.T, meta.block_n))
        sqz, sqt, sqd = (int(q.stride(0)), int(q.stride(1)), int(q.stride(2)))
        skz, skt, skd = (int(k.stride(0)), int(k.stride(1)), int(k.stride(2)))
        svz, svt, svd = (int(v.stride(0)), int(v.stride(1)), int(v.stride(2)))
        sdoz, sdot, sdod = (int(grad_out.stride(0)), int(grad_out.stride(1)), int(grad_out.stride(2)))
        slsez, slset = (int(lse.stride(0)), int(lse.stride(1)))
        sdeltaz, sdeltat = (int(delta.stride(0)), int(delta.stride(1)))
        sdkz, sdkt, sdkd = (int(dk.stride(0)), int(dk.stride(1)), int(dk.stride(2)))
        sdvz, sdvt, sdvd = (int(dv.stride(0)), int(dv.stride(1)), int(dv.stride(2)))
        if _TRITON_DEBUG:
            logger.info(
                f"Launching FlashAttention backward (dK/dV) kernel with q: {q.shape}, k: {k.shape}, v: {v.shape}, grad_out: {grad_out.shape}, lse: {lse.shape}, delta: {delta.shape}, dk: {dk.shape}, dv: {dv.shape}"
            )
        kdkv[grid](
            q, k, v, grad_out, lse, delta, dk, dv, meta.seed,
            stride_qz=sqz, stride_qt=sqt, stride_qd=sqd,
            stride_kz=skz, stride_kt=skt, stride_kd=skd,
            stride_vz=svz, stride_vt=svt, stride_vd=svd,
            stride_doz=sdoz, stride_dot=sdot, stride_dod=sdod,
            stride_lsez=slsez, stride_lset=slset,
            stride_deltaz=sdeltaz, stride_deltat=sdeltat,
            stride_dkz=sdkz, stride_dkt=sdkt, stride_dkd=sdkd,
            stride_dvz=sdvz, stride_dvt=sdvt, stride_dvd=sdvd,
            Z=meta.Z, T=meta.T, D=meta.D,
            SM_SCALE=float(meta.scale), CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n, BLOCK_D=meta.block_d,
            num_warps=4,
        )
        return dk, dv

    def _launch_dq(self, *, q: Tensor, k: Tensor, v: Tensor, grad_out: Tensor, lse: Tensor, delta: Tensor, meta: _FlashMeta) -> Tensor:
        self._require(flash_attn_bwd_dq is not None, msg="FlashAttention backward (dQ) kernel is unavailable.")
        kdq: Any = flash_attn_bwd_dq
        dq = torch.empty_like(q, dtype=torch.float32)
        grid = (meta.Z, self._cdiv(meta.T, meta.block_m))
        sqz, sqt, sqd = (int(q.stride(0)), int(q.stride(1)), int(q.stride(2)))
        skz, skt, skd = (int(k.stride(0)), int(k.stride(1)), int(k.stride(2)))
        svz, svt, svd = (int(v.stride(0)), int(v.stride(1)), int(v.stride(2)))
        sdoz, sdot, sdod = (int(grad_out.stride(0)), int(grad_out.stride(1)), int(grad_out.stride(2)))
        slsez, slset = (int(lse.stride(0)), int(lse.stride(1)))
        sdeltaz, sdeltat = (int(delta.stride(0)), int(delta.stride(1)))
        sdqz, sdqt, sdqd = (int(dq.stride(0)), int(dq.stride(1)), int(dq.stride(2)))
        if _TRITON_DEBUG:
            logger.info(
                f"Launching FlashAttention backward (dQ) kernel with q: {q.shape}, k: {k.shape}, v: {v.shape}, grad_out: {grad_out.shape}, lse: {lse.shape}, delta: {delta.shape}, dq: {dq.shape}"
            )
        kdq[grid](
            q, k, v, grad_out, lse, delta, dq, meta.seed,
            stride_qz=sqz, stride_qt=sqt, stride_qd=sqd,
            stride_kz=skz, stride_kt=skt, stride_kd=skd,
            stride_vz=svz, stride_vt=svt, stride_vd=svd,
            stride_doz=sdoz, stride_dot=sdot, stride_dod=sdod,
            stride_lsez=slsez, stride_lset=slset,
            stride_deltaz=sdeltaz, stride_deltat=sdeltat,
            stride_dqz=sdqz, stride_dqt=sdqt, stride_dqd=sdqd,
            Z=meta.Z, T=meta.T, D=meta.D,
            SM_SCALE=float(meta.scale), CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n, BLOCK_D=meta.block_d,
            num_warps=4,
        )
        return dq

    def backward(self, *, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, grad_out: Tensor, meta: _FlashMeta) -> tuple[Tensor, Tensor, Tensor]:
        self._require(grad_out.device == q.device, msg="FlashAttention backward requires grad_out on same device as q.")
        self._require(grad_out.dtype == q.dtype, msg="FlashAttention backward requires grad_out dtype to match q dtype.")
        self._require(grad_out.is_contiguous(), msg="FlashAttention backward requires contiguous grad_out.")
        B, H, T, D = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
        Z = B * H
        q2 = q.reshape(Z, T, D).contiguous()
        k2 = k.reshape(Z, T, D).contiguous()
        v2 = v.reshape(Z, T, D).contiguous()
        o2 = out.reshape(Z, T, D).contiguous()
        do2 = grad_out.reshape(Z, T, D).contiguous()
        delta = self._launch_preprocess(out=o2, grad_out=do2, meta=meta)
        dk, dv = self._launch_dkv(q=q2, k=k2, v=v2, grad_out=do2, lse=lse, delta=delta, meta=meta)
        dq = self._launch_dq(q=q2, k=k2, v=v2, grad_out=do2, lse=lse, delta=delta, meta=meta)
        if _TRITON_DEBUG:
            logger.info(f"Backward completed with dq: {dq.shape}, dk: {dk.shape}, dv: {dv.shape}")
        return dq.to(dtype=q.dtype).reshape_as(q), dk.to(dtype=k.dtype).reshape_as(k), dv.to(dtype=v.dtype).reshape_as(v)


_FLASH_IMPL = _FlashAttentionTriton()


class _FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        dropout_p: float,
        seed: int,
    ) -> Tensor:
        impl = _FLASH_IMPL
        out, lse, meta = impl.forward(
            q=q,
            k=k,
            v=v,
            causal=bool(causal),
            scale=float(scale),
            dropout_p=float(dropout_p),
            seed=int(seed),
        )
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_out: Tensor,
    ) -> tuple[Tensor | None, ...]:
        q, k, v, out, lse = ctx.saved_tensors
        impl: _FlashAttentionTriton = ctx.impl  # type: ignore[attr-defined]
        meta: _FlashMeta = ctx.meta  # type: ignore[attr-defined]
        dq, dk, dv = impl.backward(q=q, k=k, v=v, out=out, lse=lse, grad_out=grad_out.contiguous(), meta=meta)
        return (dq, dk, dv, None, None, None, None)


class FlashAttention:
    """FlashAttention training kernel.

    Use `.run(...)` to execute a fused full-sequence attention on CUDA.
    """

    def run(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        dropout_p: float = 0.0,
        seed: int | None = None,
    ) -> Tensor:
        """Run FlashAttention.

        Args:
            q/k/v: (B, H, T, D), contiguous
            causal: Whether to apply causal masking
            scale: Softmax scaling applied to QK^T
        """
        seed_i = int(torch.seed()) if seed is None else int(seed)
        y = _FlashAttnFn.apply(q, k, v, bool(causal), float(scale), float(dropout_p), seed_i)
        if not isinstance(y, torch.Tensor):
            raise TypeError("FlashAttention returned a non-tensor output")
        return y
