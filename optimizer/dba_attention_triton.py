"""CUDA/Triton DBA (decoupled) full-sequence attention forward+backward.

This is the training kernel for decoupled attention:
  logits = (q_sem @ k_sem^T) * sem_scale + (q_geo @ k_geo^T) * geo_scale

Contract:
- Full-sequence attention (no KV cache)
- Optional causal masking
- No arbitrary attention masks in the fused CUDA path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from caramba.optimizer.runtime import triton_supported


# Keep this module importable on non-CUDA installs (e.g., MPS-only laptops).
# The *symbols* exist, but kernels stay as None unless Triton is present.
dba_attn_fwd: object | None = None
dba_attn_bwd_preprocess: object | None = None
dba_attn_bwd_dkv: object | None = None
dba_attn_bwd_dq: object | None = None

if not TYPE_CHECKING and triton_supported():
    from caramba.optimizer.dba_attention_triton_kernels_bwd import (
        dba_attn_bwd_dkv,
        dba_attn_bwd_dq,
        dba_attn_bwd_preprocess,
    )
    from caramba.optimizer.dba_attention_triton_kernels_fwd import dba_attn_fwd


@dataclass(frozen=True, slots=True)
class _DBAMeta:
    Z: int
    T: int
    D_sem: int
    D_geo: int
    D_v: int
    causal: bool
    sem_scale: float
    geo_scale: float
    dropout_p: float
    seed: int
    block_m: int
    block_n: int
    block_dsem: int
    block_dgeo: int
    block_dv: int


class _DBAAttentionTriton:
    """DBA attention on CUDA via Triton kernels."""

    def _require(self, cond: bool, *, msg: str) -> None:
        if not cond:
            raise RuntimeError(msg)

    def _cdiv(self, n: int, d: int) -> int:
        return (int(n) + int(d) - 1) // int(d)

    def _validate(self, *, q_sem: Tensor, q_geo: Tensor, k_sem: Tensor, k_geo: Tensor, v: Tensor) -> tuple[int, int, int, int, int, int]:
        self._require(q_sem.device.type == "cuda", msg="DBA attention Triton requires CUDA tensors.")
        self._require(
            q_geo.device == q_sem.device and k_sem.device == q_sem.device and k_geo.device == q_sem.device and v.device == q_sem.device,
            msg="DBA attention requires all tensors on the same CUDA device.",
        )
        self._require(q_sem.ndim == 4 and q_geo.ndim == 4, msg="q_sem/q_geo must be (B,H,T,D).")
        self._require(k_sem.ndim == 4 and k_geo.ndim == 4, msg="k_sem/k_geo must be (B,H,T,D).")
        self._require(v.ndim == 4, msg="v must be (B,H,T,Dv).")
        B, H, T, D_sem = (int(q_sem.shape[0]), int(q_sem.shape[1]), int(q_sem.shape[2]), int(q_sem.shape[3]))
        self._require(q_geo.shape[:3] == (B, H, T), msg="q_geo must match q_sem on (B,H,T).")
        self._require(k_sem.shape[:3] == (B, H, T), msg="k_sem must match q_sem on (B,H,T).")
        self._require(k_geo.shape[:3] == (B, H, T), msg="k_geo must match q_sem on (B,H,T).")
        self._require(v.shape[:3] == (B, H, T), msg="v must match q_sem on (B,H,T).")
        D_geo = int(q_geo.shape[3])
        self._require(int(k_sem.shape[3]) == D_sem, msg="k_sem head dim must match q_sem head dim.")
        self._require(int(k_geo.shape[3]) == D_geo, msg="k_geo head dim must match q_geo head dim.")
        D_v = int(v.shape[3])
        self._require(q_sem.dtype == q_geo.dtype == k_sem.dtype == k_geo.dtype == v.dtype, msg="All DBA tensors must share dtype.")
        self._require(q_sem.dtype in (torch.float16, torch.bfloat16, torch.float32), msg="DBA attention supports fp16/bf16/fp32.")
        self._require(
            q_sem.is_contiguous() and q_geo.is_contiguous() and k_sem.is_contiguous() and k_geo.is_contiguous() and v.is_contiguous(),
            msg="DBA attention requires contiguous inputs.",
        )
        self._require(D_sem <= 128 and D_geo <= 128, msg=f"DBA attention currently supports sem/geo head dims <= 128 (got {D_sem}, {D_geo}).")
        self._require(D_v <= 256, msg=f"DBA attention currently supports v_head_dim <= 256 (got {D_v}).")
        return B, H, T, D_sem, D_geo, D_v

    def _meta(
        self,
        *,
        Z: int,
        T: int,
        D_sem: int,
        D_geo: int,
        D_v: int,
        causal: bool,
        sem_scale: float,
        geo_scale: float,
        dropout_p: float,
        seed: int,
    ) -> _DBAMeta:
        return _DBAMeta(
            Z=Z,
            T=T,
            D_sem=D_sem,
            D_geo=D_geo,
            D_v=D_v,
            causal=bool(causal),
            sem_scale=float(sem_scale),
            geo_scale=float(geo_scale),
            dropout_p=float(dropout_p),
            seed=int(seed),
            block_m=64,
            block_n=64,
            block_dsem=32 if int(D_sem) <= 32 else (64 if int(D_sem) <= 64 else 128),
            block_dgeo=32 if int(D_geo) <= 32 else (64 if int(D_geo) <= 64 else 128),
            block_dv=64 if int(D_v) <= 64 else (128 if int(D_v) <= 128 else 256),
        )

    def forward(
        self,
        *,
        q_sem: Tensor,
        q_geo: Tensor,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        causal: bool,
        sem_scale: float,
        geo_scale: float,
        dropout_p: float,
        seed: int,
    ) -> tuple[Tensor, Tensor, _DBAMeta]:
        B, H, T, D_sem, D_geo, D_v = self._validate(q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, v=v)
        Z = B * H
        meta = self._meta(
            Z=Z,
            T=T,
            D_sem=D_sem,
            D_geo=D_geo,
            D_v=D_v,
            causal=bool(causal),
            sem_scale=float(sem_scale),
            geo_scale=float(geo_scale),
            dropout_p=float(dropout_p),
            seed=int(seed),
        )
        q_sem2 = q_sem.reshape(Z, T, D_sem)
        q_geo2 = q_geo.reshape(Z, T, D_geo)
        k_sem2 = k_sem.reshape(Z, T, D_sem)
        k_geo2 = k_geo.reshape(Z, T, D_geo)
        v2 = v.reshape(Z, T, D_v)

        out = torch.empty((Z, T, D_v), device=q_sem.device, dtype=torch.float32)
        lse = torch.empty((Z, T), device=q_sem.device, dtype=torch.float32)
        self._require(dba_attn_fwd is not None, msg="DBA attention forward kernel is unavailable.")
        kf: Any = dba_attn_fwd
        grid = (Z, self._cdiv(T, meta.block_m))
        kf[grid](
            q_sem2,
            q_geo2,
            k_sem2,
            k_geo2,
            v2,
            out,
            lse,
            meta.seed,
            stride_qsz=q_sem2.stride(0), stride_qst=q_sem2.stride(1), stride_qsd=q_sem2.stride(2),
            stride_qgz=q_geo2.stride(0), stride_qgt=q_geo2.stride(1), stride_qgd=q_geo2.stride(2),
            stride_ksz=k_sem2.stride(0), stride_kst=k_sem2.stride(1), stride_ksd=k_sem2.stride(2),
            stride_kgz=k_geo2.stride(0), stride_kgt=k_geo2.stride(1), stride_kgd=k_geo2.stride(2),
            stride_vz=v2.stride(0), stride_vt=v2.stride(1), stride_vd=v2.stride(2),
            stride_oz=out.stride(0), stride_ot=out.stride(1), stride_od=out.stride(2),
            stride_lsez=lse.stride(0), stride_lset=lse.stride(1),
            Z=Z, T=T, D_SEM=D_sem, D_GEO=D_geo, D_V=D_v,
            SEM_SCALE=float(meta.sem_scale), GEO_SCALE=float(meta.geo_scale),
            CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n,
            BLOCK_DSEM=meta.block_dsem, BLOCK_DGEO=meta.block_dgeo, BLOCK_DV=meta.block_dv,
            num_warps=4,
        )
        return out.to(dtype=q_sem.dtype).reshape(B, H, T, D_v), lse, meta

    def backward(
        self,
        *,
        q_sem: Tensor,
        q_geo: Tensor,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        out: Tensor,
        lse: Tensor,
        grad_out: Tensor,
        meta: _DBAMeta,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._require(grad_out.device == q_sem.device, msg="DBA backward requires grad_out on same device.")
        self._require(grad_out.dtype == q_sem.dtype, msg="DBA backward requires grad_out dtype to match inputs.")
        B, H, T = int(q_sem.shape[0]), int(q_sem.shape[1]), int(q_sem.shape[2])
        Z = B * H

        q_sem2 = q_sem.reshape(Z, T, meta.D_sem).contiguous()
        q_geo2 = q_geo.reshape(Z, T, meta.D_geo).contiguous()
        k_sem2 = k_sem.reshape(Z, T, meta.D_sem).contiguous()
        k_geo2 = k_geo.reshape(Z, T, meta.D_geo).contiguous()
        v2 = v.reshape(Z, T, meta.D_v).contiguous()
        out2 = out.reshape(Z, T, meta.D_v).contiguous()
        do2 = grad_out.reshape(Z, T, meta.D_v).contiguous()

        self._require(dba_attn_bwd_preprocess is not None, msg="DBA preprocess kernel is unavailable.")
        self._require(dba_attn_bwd_dkv is not None and dba_attn_bwd_dq is not None, msg="DBA backward kernels are unavailable.")
        kp: Any = dba_attn_bwd_preprocess
        kdkv: Any = dba_attn_bwd_dkv
        kdq: Any = dba_attn_bwd_dq

        delta = torch.empty((Z, T), device=q_sem.device, dtype=torch.float32)
        grid_m = (Z, self._cdiv(T, meta.block_m))
        kp[grid_m](
            out2,
            do2,
            delta,
            stride_oz=out2.stride(0), stride_ot=out2.stride(1), stride_od=out2.stride(2),
            stride_doz=do2.stride(0), stride_dot=do2.stride(1), stride_dod=do2.stride(2),
            stride_deltaz=delta.stride(0), stride_deltat=delta.stride(1),
            Z=Z, T=T, D_V=meta.D_v,
            BLOCK_M=meta.block_m, BLOCK_DV=meta.block_dv,
            num_warps=4,
        )

        dk_sem = torch.empty((Z, T, meta.D_sem), device=q_sem.device, dtype=torch.float32)
        dk_geo = torch.empty((Z, T, meta.D_geo), device=q_sem.device, dtype=torch.float32)
        dv = torch.empty((Z, T, meta.D_v), device=q_sem.device, dtype=torch.float32)
        grid_n = (Z, self._cdiv(T, meta.block_n))
        kdkv[grid_n](
            q_sem2, q_geo2, k_sem2, k_geo2, v2, do2, lse, delta,
            dk_sem, dk_geo, dv,
            meta.seed,
            stride_qsz=q_sem2.stride(0), stride_qst=q_sem2.stride(1), stride_qsd=q_sem2.stride(2),
            stride_qgz=q_geo2.stride(0), stride_qgt=q_geo2.stride(1), stride_qgd=q_geo2.stride(2),
            stride_ksz=k_sem2.stride(0), stride_kst=k_sem2.stride(1), stride_ksd=k_sem2.stride(2),
            stride_kgz=k_geo2.stride(0), stride_kgt=k_geo2.stride(1), stride_kgd=k_geo2.stride(2),
            stride_vz=v2.stride(0), stride_vt=v2.stride(1), stride_vd=v2.stride(2),
            stride_doz=do2.stride(0), stride_dot=do2.stride(1), stride_dod=do2.stride(2),
            stride_lsez=lse.stride(0), stride_lset=lse.stride(1),
            stride_deltaz=delta.stride(0), stride_deltat=delta.stride(1),
            stride_dksz=dk_sem.stride(0), stride_dkst=dk_sem.stride(1), stride_dksd=dk_sem.stride(2),
            stride_dkgz=dk_geo.stride(0), stride_dkgt=dk_geo.stride(1), stride_dkgd=dk_geo.stride(2),
            stride_dvz=dv.stride(0), stride_dvt=dv.stride(1), stride_dvd=dv.stride(2),
            Z=Z, T=T, D_SEM=meta.D_sem, D_GEO=meta.D_geo, D_V=meta.D_v,
            SEM_SCALE=float(meta.sem_scale), GEO_SCALE=float(meta.geo_scale),
            CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n,
            BLOCK_DSEM=meta.block_dsem, BLOCK_DGEO=meta.block_dgeo, BLOCK_DV=meta.block_dv,
            num_warps=4,
        )

        dq_sem = torch.empty((Z, T, meta.D_sem), device=q_sem.device, dtype=torch.float32)
        dq_geo = torch.empty((Z, T, meta.D_geo), device=q_sem.device, dtype=torch.float32)
        kdq[grid_m](
            q_sem2, q_geo2, k_sem2, k_geo2, v2, do2, lse, delta,
            dq_sem, dq_geo,
            meta.seed,
            stride_qsz=q_sem2.stride(0), stride_qst=q_sem2.stride(1), stride_qsd=q_sem2.stride(2),
            stride_qgz=q_geo2.stride(0), stride_qgt=q_geo2.stride(1), stride_qgd=q_geo2.stride(2),
            stride_ksz=k_sem2.stride(0), stride_kst=k_sem2.stride(1), stride_ksd=k_sem2.stride(2),
            stride_kgz=k_geo2.stride(0), stride_kgt=k_geo2.stride(1), stride_kgd=k_geo2.stride(2),
            stride_vz=v2.stride(0), stride_vt=v2.stride(1), stride_vd=v2.stride(2),
            stride_doz=do2.stride(0), stride_dot=do2.stride(1), stride_dod=do2.stride(2),
            stride_lsez=lse.stride(0), stride_lset=lse.stride(1),
            stride_deltaz=delta.stride(0), stride_deltat=delta.stride(1),
            stride_dqsz=dq_sem.stride(0), stride_dqst=dq_sem.stride(1), stride_dqsd=dq_sem.stride(2),
            stride_dqgz=dq_geo.stride(0), stride_dqgt=dq_geo.stride(1), stride_dqgd=dq_geo.stride(2),
            Z=Z, T=T, D_SEM=meta.D_sem, D_GEO=meta.D_geo, D_V=meta.D_v,
            SEM_SCALE=float(meta.sem_scale), GEO_SCALE=float(meta.geo_scale),
            CAUSAL=int(meta.causal),
            USE_DROPOUT=int(meta.dropout_p > 0.0),
            DROPOUT_P=float(meta.dropout_p),
            BLOCK_M=meta.block_m, BLOCK_N=meta.block_n,
            BLOCK_DSEM=meta.block_dsem, BLOCK_DGEO=meta.block_dgeo, BLOCK_DV=meta.block_dv,
            num_warps=4,
        )

        return (
            dq_sem.to(dtype=q_sem.dtype).reshape_as(q_sem),
            dq_geo.to(dtype=q_geo.dtype).reshape_as(q_geo),
            dk_sem.to(dtype=k_sem.dtype).reshape_as(k_sem),
            dk_geo.to(dtype=k_geo.dtype).reshape_as(k_geo),
            dv.to(dtype=v.dtype).reshape_as(v),
        )


class _DBAAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_sem: Tensor,
        q_geo: Tensor,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        causal: bool,
        sem_scale: float,
        geo_scale: float,
        dropout_p: float,
        seed: int,
    ) -> Tensor:
        impl = _DBAAttentionTriton()
        out, lse, meta = impl.forward(
            q_sem=q_sem,
            q_geo=q_geo,
            k_sem=k_sem,
            k_geo=k_geo,
            v=v,
            causal=bool(causal),
            sem_scale=float(sem_scale),
            geo_scale=float(geo_scale),
            dropout_p=float(dropout_p),
            seed=int(seed),
        )
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(q_sem, q_geo, k_sem, k_geo, v, out, lse)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_out: Tensor,
    ) -> tuple[Tensor | None, ...]:
        q_sem, q_geo, k_sem, k_geo, v, out, lse = ctx.saved_tensors
        impl: _DBAAttentionTriton = ctx.impl  # type: ignore[attr-defined]
        meta: _DBAMeta = ctx.meta  # type: ignore[attr-defined]
        dq_sem, dq_geo, dk_sem, dk_geo, dv = impl.backward(
            q_sem=q_sem,
            q_geo=q_geo,
            k_sem=k_sem,
            k_geo=k_geo,
            v=v,
            out=out,
            lse=lse,
            grad_out=grad_out.contiguous(),
            meta=meta,
        )
        return (dq_sem, dq_geo, dk_sem, dk_geo, dv, None, None, None, None, None)


class DecoupledAttentionTraining:
    """DBA full-sequence attention training kernel (CUDA Triton)."""

    def run(
        self,
        *,
        q_sem: Tensor,
        q_geo: Tensor,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        causal: bool,
        sem_scale: float,
        geo_scale: float,
        dropout_p: float = 0.0,
        seed: int | None = None,
    ) -> Tensor:
        seed_i = int(torch.seed()) if seed is None else int(seed)
        y = _DBAAttnFn.apply(
            q_sem,
            q_geo,
            k_sem,
            k_geo,
            v,
            bool(causal),
            float(sem_scale),
            float(geo_scale),
            float(dropout_p),
            seed_i,
        )
        if not isinstance(y, torch.Tensor):
            raise TypeError("DecoupledAttentionTraining returned a non-tensor output")
        return y
