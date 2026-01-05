"""Fused decode path for decoupled caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from caramba.console import logger

if TYPE_CHECKING:
    from caramba.cache.decoupled import DecoupledLayerKVCache


class DecoupledDecode:
    """Fused DBA decode dispatch for CUDA (Triton) and MPS (Metal)."""

    def run(
        self,
        *,
        q_sem: Tensor,
        q_geo: Tensor,
        cache: "DecoupledLayerKVCache",
        n_heads: int,
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        sem_scale: float,
        geo_scale: float,
        decode_block: int,
        null_enabled: bool,
        null_kv,
    ) -> Tensor:
        if q_sem.device.type == "cuda":
            from caramba.optimizer.fused_attention import (
                fused_decode_available,
                fused_decode_decoupled_q4q8q4,
                fused_decode_decoupled_q4q8q4_2pass,
            )

            if not fused_decode_available(cache, "cuda"):
                raise RuntimeError(
                    "Fused DBA decode is required for CUDA decode, but is unavailable for this cache/device.\n"
                    "Ensure Triton is installed and the decoupled KV cache uses q4_0/q8_0/q4_0 with qblock=32.\n"
                )

            ksn = kgn = vn = None
            if null_enabled:
                ksn, kgn, vn = null_kv(B=int(q_sem.size(0)), dtype=q_sem.dtype, device=q_sem.device)

            cache_len = int(cache.pos)
            if cache_len > 4 * int(decode_block):
                return fused_decode_decoupled_q4q8q4_2pass(
                    q_sem=q_sem,
                    q_geo=q_geo,
                    cache=cache,
                    n_heads=int(n_heads),
                    sem_head_dim=int(sem_head_dim),
                    geo_head_dim=int(geo_head_dim),
                    v_head_dim=int(v_head_dim),
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                    decode_block=int(decode_block),
                    k_sem_null=ksn,
                    k_geo_null=kgn,
                    v_null=vn,
                )

            return fused_decode_decoupled_q4q8q4(
                q_sem=q_sem,
                q_geo=q_geo,
                cache=cache,
                n_heads=int(n_heads),
                sem_head_dim=int(sem_head_dim),
                geo_head_dim=int(geo_head_dim),
                v_head_dim=int(v_head_dim),
                sem_scale=float(sem_scale),
                geo_scale=float(geo_scale),
                decode_block=int(decode_block),
                k_sem_null=ksn,
                k_geo_null=kgn,
                v_null=vn,
            )

        if q_sem.device.type == "mps":
            from caramba.optimizer.metal import dba_decode_fp16, metal_dba_decode_available

            if not metal_dba_decode_available():
                raise RuntimeError("Metal DBA decode kernel is unavailable on this runtime.")

            if not (cache.k_sem.kind == "fp16" and cache.k_geo.kind == "fp16" and cache.v.kind == "fp16"):
                raise RuntimeError(
                    "Metal DBA decode requires fp16 KV caches on MPS.\n"
                    f"Got kinds: k_sem={cache.k_sem.kind}, k_geo={cache.k_geo.kind}, v={cache.v.kind}\n"
                )
            if cache.k_sem.buf is None or cache.k_geo.buf is None or cache.v.buf is None:
                raise RuntimeError("fp16 KV cache buffers are not initialized")

            S = int(cache.pos)
            k_sem_all = cache.k_sem.buf.narrow(1, 0, S)
            k_geo_all = cache.k_geo.buf.narrow(1, 0, S)
            v_all = cache.v.buf.narrow(1, 0, S)

            ksn = kgn = vn = None
            if null_enabled:
                ksn, kgn, vn = null_kv(B=int(q_sem.size(0)), dtype=q_sem.dtype, device=q_sem.device)

            return dba_decode_fp16(
                q_sem=q_sem,
                q_geo=q_geo,
                k_sem=k_sem_all,
                k_geo=k_geo_all,
                v=v_all,
                k_sem_null=ksn,
                k_geo_null=kgn,
                v_null=vn,
                sem_scale=float(sem_scale),
                geo_scale=float(geo_scale),
                verbose_build=False,
            )

        logger.error(f"Unsupported device for fused DBA decode: {q_sem.device.type}")
        raise RuntimeError(f"Unsupported device for fused DBA decode: {q_sem.device.type}")

