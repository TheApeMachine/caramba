"""Triton CUDA kernels for FlashAttention backward.

The backward follows the standard FlashAttention derivation:
- Forward stores per-query LSE and output.
- Backward recomputes blockwise attention probabilities from Q,K and LSE and
  accumulates dV/dK/dQ without materializing the full attention matrix.

This kernel is used by `optimizer/flash_attention_triton.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.kernel.runtime import triton_supported

if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:
        __all__ = ["flash_attn_bwd_dkv", "flash_attn_bwd_dq", "flash_attn_bwd_preprocess"]


        @triton.jit
        def flash_attn_bwd_preprocess(
            out_ptr,
            do_ptr,
            delta_ptr,
            stride_oz: tl.constexpr,
            stride_ot: tl.constexpr,
            stride_od: tl.constexpr,
            stride_doz: tl.constexpr,
            stride_dot: tl.constexpr,
            stride_dod: tl.constexpr,
            stride_deltaz: tl.constexpr,
            stride_deltat: tl.constexpr,
            Z: tl.constexpr,
            T: tl.constexpr,
            D: tl.constexpr,
            BLOCK_M: tl.constexpr = 64,
            BLOCK_D: tl.constexpr = 128,
        ):
            """Compute delta = sum_d(out * dOut) per query token (float32)."""
            pid_z = tl.program_id(0)
            pid_m = tl.program_id(1)
            z = pid_z
            m_start = pid_m * BLOCK_M

            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_D)
            m_mask = offs_m < T
            d_mask = offs_d < D

            o = tl.load(
                out_ptr + z * stride_oz + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            delta = tl.sum(o * do, axis=1)
            tl.store(
                delta_ptr + z * stride_deltaz + offs_m * stride_deltat,
                delta,
                mask=m_mask,
            )


        @triton.jit
        def flash_attn_bwd_dkv(
            q_ptr,
            k_ptr,
            v_ptr,
            do_ptr,
            lse_ptr,
            delta_ptr,
            dk_ptr,
            dv_ptr,
            seed: tl.uint32,
            stride_qz: tl.constexpr,
            stride_qt: tl.constexpr,
            stride_qd: tl.constexpr,
            stride_kz: tl.constexpr,
            stride_kt: tl.constexpr,
            stride_kd: tl.constexpr,
            stride_vz: tl.constexpr,
            stride_vt: tl.constexpr,
            stride_vd: tl.constexpr,
            stride_doz: tl.constexpr,
            stride_dot: tl.constexpr,
            stride_dod: tl.constexpr,
            stride_lsez: tl.constexpr,
            stride_lset: tl.constexpr,
            stride_deltaz: tl.constexpr,
            stride_deltat: tl.constexpr,
            stride_dkz: tl.constexpr,
            stride_dkt: tl.constexpr,
            stride_dkd: tl.constexpr,
            stride_dvz: tl.constexpr,
            stride_dvt: tl.constexpr,
            stride_dvd: tl.constexpr,
            Z: tl.constexpr,
            T: tl.constexpr,
            D: tl.constexpr,
            SM_SCALE: tl.constexpr,
            CAUSAL: tl.constexpr,
            USE_DROPOUT: tl.constexpr,
            DROPOUT_P: tl.constexpr,
            BLOCK_M: tl.constexpr = 64,
            BLOCK_N: tl.constexpr = 64,
            BLOCK_D: tl.constexpr = 128,
        ):
            """Compute dK and dV for a (Z, T, D) attention.

            Parallelized over (z, n_block), where each program computes gradients for
            one block of keys (BLOCK_N tokens).
            """
            pid_z = tl.program_id(0)
            pid_n = tl.program_id(1)
            z = pid_z
            n_start = pid_n * BLOCK_N

            offs_n = n_start + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_D)
            n_mask = offs_n < T
            d_mask = offs_d < D

            k = tl.load(
                k_ptr + z * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
            dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

            # Iterate over queries in blocks of BLOCK_M.
            for m_start in range(0, T, BLOCK_M):
                offs_m = m_start + tl.arange(0, BLOCK_M)
                m_mask = offs_m < T

                q = tl.load(
                    q_ptr + z * stride_qz + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=m_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                do = tl.load(
                    do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod,
                    mask=m_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )

                lse = tl.load(lse_ptr + z * stride_lsez + offs_m * stride_lset, mask=m_mask, other=-float("inf")).to(
                    tl.float32
                )
                delta = tl.load(
                    delta_ptr + z * stride_deltaz + offs_m * stride_deltat, mask=m_mask, other=0.0
                ).to(tl.float32)

                qk = tl.dot(q, tl.trans(k)) * SM_SCALE
                if CAUSAL:
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    qk = tl.where(k_pos <= q_pos, qk, -float("inf"))

                p = tl.exp(qk - lse[:, None])

                if USE_DROPOUT:
                    keep_prob = 1.0 - float(DROPOUT_P)
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    rng = tl.rand(seed + tl.full([], z, tl.uint32), q_pos * T + k_pos)
                    keep = rng < keep_prob
                    w = p * keep.to(tl.float32) * (1.0 / keep_prob)
                else:
                    w = p

                # dV: P^T @ dO
                dv += tl.dot(tl.trans(w.to(do.dtype)), do)

                # dK: (P * (dP))^T @ Q, where dP = (dO @ V^T) and softmax jacobian uses (dP - delta)
                dp = tl.dot(do, tl.trans(v))
                ds = (w * (dp - delta[:, None])) * SM_SCALE
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

            tl.store(
                dk_ptr + z * stride_dkz + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd,
                dk,
                mask=n_mask[:, None] & d_mask[None, :],
            )
            tl.store(
                dv_ptr + z * stride_dvz + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd,
                dv,
                mask=n_mask[:, None] & d_mask[None, :],
            )


        @triton.jit
        def flash_attn_bwd_dq(
            q_ptr,
            k_ptr,
            v_ptr,
            do_ptr,
            lse_ptr,
            delta_ptr,
            dq_ptr,
            seed: tl.uint32,
            stride_qz: tl.constexpr,
            stride_qt: tl.constexpr,
            stride_qd: tl.constexpr,
            stride_kz: tl.constexpr,
            stride_kt: tl.constexpr,
            stride_kd: tl.constexpr,
            stride_vz: tl.constexpr,
            stride_vt: tl.constexpr,
            stride_vd: tl.constexpr,
            stride_doz: tl.constexpr,
            stride_dot: tl.constexpr,
            stride_dod: tl.constexpr,
            stride_lsez: tl.constexpr,
            stride_lset: tl.constexpr,
            stride_deltaz: tl.constexpr,
            stride_deltat: tl.constexpr,
            stride_dqz: tl.constexpr,
            stride_dqt: tl.constexpr,
            stride_dqd: tl.constexpr,
            Z: tl.constexpr,
            T: tl.constexpr,
            D: tl.constexpr,
            SM_SCALE: tl.constexpr,
            CAUSAL: tl.constexpr,
            USE_DROPOUT: tl.constexpr,
            DROPOUT_P: tl.constexpr,
            BLOCK_M: tl.constexpr = 64,
            BLOCK_N: tl.constexpr = 64,
            BLOCK_D: tl.constexpr = 128,
        ):
            """Compute dQ for a (Z, T, D) attention.

            Parallelized over (z, m_block), where each program computes gradients for
            one block of queries (BLOCK_M tokens).
            """
            pid_z = tl.program_id(0)
            pid_m = tl.program_id(1)
            z = pid_z
            m_start = pid_m * BLOCK_M

            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_D)
            m_mask = offs_m < T
            d_mask = offs_d < D

            q = tl.load(
                q_ptr + z * stride_qz + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            do = tl.load(
                do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            lse = tl.load(lse_ptr + z * stride_lsez + offs_m * stride_lset, mask=m_mask, other=-float("inf")).to(tl.float32)
            delta = tl.load(
                delta_ptr + z * stride_deltaz + offs_m * stride_deltat, mask=m_mask, other=0.0
            ).to(tl.float32)

            dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

            for n_start in range(0, T, BLOCK_N):
                offs_n = n_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < T

                k = tl.load(
                    k_ptr + z * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )

                qk = tl.dot(q, tl.trans(k)) * SM_SCALE
                if CAUSAL:
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    qk = tl.where(k_pos <= q_pos, qk, -float("inf"))
                p = tl.exp(qk - lse[:, None])

                if USE_DROPOUT:
                    keep_prob = 1.0 - float(DROPOUT_P)
                    q_pos = offs_m[:, None]
                    k_pos = offs_n[None, :]
                    rng = tl.rand(seed + tl.full([], z, tl.uint32), q_pos * T + k_pos)
                    keep = rng < keep_prob
                    w = p * keep.to(tl.float32) * (1.0 / keep_prob)
                else:
                    w = p

                dp = tl.dot(do, tl.trans(v))
                ds = (w * (dp - delta[:, None])) * SM_SCALE
                dq += tl.dot(ds.to(k.dtype), k)

            tl.store(
                dq_ptr + z * stride_dqz + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd,
                dq,
                mask=m_mask[:, None] & d_mask[None, :],
            )
