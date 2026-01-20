"""Fused DBA decode

Single-token decoding is a tight loop; this module dispatches to fused kernels
so per-step attention becomes bandwidth/compute efficient instead of Python
overhead dominated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from console import logger

if TYPE_CHECKING:
    from cache.decoupled import DecoupledLayerKVCache


class DecoupledDecode:
    """Fused DBA decode dispatcher

    The decode kernels are specialized for the cache layout and quantization,
    which is why decode has its own fast path instead of reusing generic SDPA.
    """

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
            from optimizer.fused_attention import (
                fused_decode_available,
                fused_decode_decoupled_q4q8q4,
                fused_decode_decoupled_q4q8q4_2pass,
            )

            if not fused_decode_available(cache, "cuda"):
                # Fallback to SDPA for fp16 caches (slower but works)
                return self._sdpa_decode_fallback(
                    q_sem=q_sem,
                    q_geo=q_geo,
                    cache=cache,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    null_enabled=null_enabled,
                    null_kv=null_kv,
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
            from optimizer.metal import dba_decode_fp16, metal_dba_decode_available

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

    def _sdpa_decode_fallback(
        self,
        *,
        q_sem: Tensor,
        q_geo: Tensor,
        cache: "DecoupledLayerKVCache",
        sem_scale: float,
        geo_scale: float,
        null_enabled: bool,
        null_kv,
    ) -> Tensor:
        """Fallback SDPA-based decode for fp16 caches.

        This is slower than the fused Triton kernel but works for any cache type.
        Used when quantized caches aren't available.
        """
        import torch.nn.functional as F

        # Get cached K/V from fp16 buffers
        S = int(cache.pos)
        if cache.k_sem.buf is None or cache.k_geo.buf is None or cache.v.buf is None:
            raise RuntimeError("fp16 KV cache buffers are not initialized")

        # Shape: (B, S, n_heads * head_dim) -> need to reshape
        k_sem_all = cache.k_sem.buf.narrow(1, 0, S)  # (B, S, n_heads * sem_dim)
        k_geo_all = cache.k_geo.buf.narrow(1, 0, S)  # (B, S, n_heads * geo_dim)
        v_all = cache.v.buf.narrow(1, 0, S)  # (B, S, n_heads * v_dim)

        B = q_sem.size(0)
        n_heads = q_sem.size(1)
        sem_head_dim = q_sem.size(-1)
        geo_head_dim = q_geo.size(-1)
        v_head_dim = v_all.size(-1) // n_heads

        # Reshape to (B, n_heads, S, head_dim)
        k_sem = k_sem_all.view(B, S, n_heads, sem_head_dim).transpose(1, 2)
        k_geo = k_geo_all.view(B, S, n_heads, geo_head_dim).transpose(1, 2)
        v = v_all.view(B, S, n_heads, v_head_dim).transpose(1, 2)

        # Concatenate scaled Q and K for combined attention
        q_cat = torch.cat([q_sem * sem_scale, q_geo * geo_scale], dim=-1)  # (B, H, 1, sem+geo)
        k_cat = torch.cat([k_sem, k_geo], dim=-1)  # (B, H, S, sem+geo)

        # Handle null attention if enabled
        if null_enabled and null_kv is not None:
            ksn, kgn, vn = null_kv(B=B, dtype=q_sem.dtype, device=q_sem.device)
            k_null = torch.cat([ksn, kgn], dim=-1)  # (B, H, 1, sem+geo)
            k_cat = torch.cat([k_null, k_cat], dim=2)  # (B, H, S+1, sem+geo)
            v = torch.cat([vn, v], dim=2)  # (B, H, S+1, v_dim)

        # SDPA attention (decode is non-causal since we're attending to past)
        out = F.scaled_dot_product_attention(
            q_cat, k_cat, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )  # (B, H, 1, v_dim)

        return out

