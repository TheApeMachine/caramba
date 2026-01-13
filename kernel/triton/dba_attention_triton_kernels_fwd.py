"""Triton CUDA kernels for DBA (decoupled) full-sequence attention forward.

Implements FlashAttention-style streaming softmax for the decoupled score:

  logits = (q_sem @ k_sem^T) * sem_scale + (q_geo @ k_geo^T) * geo_scale

Unlike standard attention, Q/K head dimensions can differ from V head dimension.
Forward stores per-query log-sum-exp (LSE) for backward.
"""

from typing import TYPE_CHECKING
from caramba.kernel.runtime import triton_supported

if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:
        __all__ = ["dba_attn_fwd"]

        # Used to keep forward/backward scaling consistent when combining sem+geo logits.
        INV_SQRT2 = tl.constexpr(0.70710678)


        @triton.jit
        def dba_attn_fwd(
            q_sem_ptr,
            q_geo_ptr,
            k_sem_ptr,
            k_geo_ptr,
            v_ptr,
            out_ptr,
            lse_ptr,
            seed: tl.uint32,
            stride_qsz: tl.constexpr,
            stride_qst: tl.constexpr,
            stride_qsd: tl.constexpr,
            stride_qgz: tl.constexpr,
            stride_qgt: tl.constexpr,
            stride_qgd: tl.constexpr,
            stride_ksz: tl.constexpr,
            stride_kst: tl.constexpr,
            stride_ksd: tl.constexpr,
            stride_kgz: tl.constexpr,
            stride_kgt: tl.constexpr,
            stride_kgd: tl.constexpr,
            stride_vz: tl.constexpr,
            stride_vt: tl.constexpr,
            stride_vd: tl.constexpr,
            stride_oz: tl.constexpr,
            stride_ot: tl.constexpr,
            stride_od: tl.constexpr,
            stride_lsez: tl.constexpr,
            stride_lset: tl.constexpr,
            Z: tl.constexpr,
            T: tl.constexpr,
            D_SEM: tl.constexpr,
            D_GEO: tl.constexpr,
            D_V: tl.constexpr,
            SEM_SCALE: tl.constexpr,
            GEO_SCALE: tl.constexpr,
            CAUSAL: tl.constexpr,
            USE_DROPOUT: tl.constexpr,
            DROPOUT_P: tl.constexpr,
            BLOCK_M: tl.constexpr = 64,
            BLOCK_N: tl.constexpr = 64,
            BLOCK_DSEM: tl.constexpr = 128,
            BLOCK_DGEO: tl.constexpr = 128,
            BLOCK_DV: tl.constexpr = 256,
        ):
            """DBA FlashAttention forward for flattened (Z,T,*) tensors."""
            pid_z = tl.program_id(0)
            pid_m = tl.program_id(1)
            z = pid_z
            m_start = pid_m * BLOCK_M

            offs_m = m_start + tl.arange(0, BLOCK_M)
            m_mask = (offs_m < T).to(tl.int1)  # type: ignore

            ds = tl.arange(0, BLOCK_DSEM)
            dg = tl.arange(0, BLOCK_DGEO)
            dv = tl.arange(0, BLOCK_DV)
            ms = (ds < D_SEM).to(tl.int1)  # type: ignore
            mg = (dg < D_GEO).to(tl.int1)  # type: ignore
            mv = (dv < D_V).to(tl.int1)  # type: ignore

            q_sem = tl.load(
                q_sem_ptr + z * stride_qsz + offs_m[:, None] * stride_qst + ds[None, :] * stride_qsd,
                mask=m_mask[:, None] & ms[None, :],
                other=0.0,
            )
            q_geo = tl.load(
                q_geo_ptr + z * stride_qgz + offs_m[:, None] * stride_qgt + dg[None, :] * stride_qgd,
                mask=m_mask[:, None] & mg[None, :],
                other=0.0,
            )

            m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
            l_i = tl.zeros((BLOCK_M,), tl.float32)
            acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

            for n_start in range(0, T, BLOCK_N):
                offs_n = n_start + tl.arange(0, BLOCK_N)
                n_mask = (offs_n < T).to(tl.int1)  # type: ignore

                k_sem = tl.load(
                    k_sem_ptr + z * stride_ksz + offs_n[:, None] * stride_kst + ds[None, :] * stride_ksd,
                    mask=n_mask[:, None] & ms[None, :],
                    other=0.0,
                )
                k_geo = tl.load(
                    k_geo_ptr + z * stride_kgz + offs_n[:, None] * stride_kgt + dg[None, :] * stride_kgd,
                    mask=n_mask[:, None] & mg[None, :],
                    other=0.0,
                )

                sem = tl.dot(q_sem, tl.trans(k_sem)) * SEM_SCALE
                geo = tl.dot(q_geo, tl.trans(k_geo)) * GEO_SCALE
                logits = (sem + geo) * INV_SQRT2

                if CAUSAL:
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    logits = tl.where(k_pos <= q_pos, logits, -float("inf"))

                m_ij = tl.max(logits, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                p = tl.exp(logits - m_new[:, None])
                alpha = tl.exp(m_i - m_new)
                l_new = l_i * alpha + tl.sum(p, axis=1)

                v = tl.load(
                    v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + dv[None, :] * stride_vd,
                    mask=n_mask[:, None] & mv[None, :],
                    other=0.0,
                )
                if USE_DROPOUT:
                    keep_prob = 1.0 - float(DROPOUT_P)
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    rng = tl.rand(seed + tl.full([], z, tl.uint32), q_pos * T + k_pos)
                    keep = (rng < keep_prob).to(tl.int1)  # type: ignore
                    p_num = p * keep.to(tl.float32) * (1.0 / keep_prob)
                else:
                    p_num = p

                # Cast the softmax numerator to V's dtype so matmul can use tensor cores
                # (accumulation still happens in fp32).
                p_num = p_num.to(v.dtype)
                acc = acc * alpha[:, None] + tl.dot(p_num, v)

                m_i = m_new
                l_i = l_new

            out = acc / l_i[:, None]
            tl.store(
                out_ptr + z * stride_oz + offs_m[:, None] * stride_ot + dv[None, :] * stride_od,
                out,
                mask=m_mask[:, None] & mv[None, :],
            )
            lse = m_i + tl.log(l_i)
            tl.store(lse_ptr + z * stride_lsez + offs_m * stride_lset, lse, mask=m_mask)
