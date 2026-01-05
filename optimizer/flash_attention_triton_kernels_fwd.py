"""Triton CUDA kernels for FlashAttention forward.

Implements a FlashAttention-style streaming softmax:
- Computes attention without materializing the full (Tq x Tk) matrix.
- Stores per-query log-sum-exp (LSE) for backward.

This kernel is used by `optimizer/flash_attention_triton.py`.
"""

import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

__all__ = ["flash_attn_fwd"]


@triton.jit
def flash_attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
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
    stride_oz: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lsez: tl.constexpr,
    stride_lset: tl.constexpr,
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
    """FlashAttention forward for (Z,T,D) tensors.

    Shapes:
      q, k, v: (Z, T, D)
      out:     (Z, T, D)
      lse:     (Z, T)      (float32)
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
    ).to(tl.float32)

    # Running online softmax state.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Loop over keys/values in blocks of BLOCK_N.
    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k = tl.load(
            k_ptr + z * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k))
        qk *= SM_SCALE

        if CAUSAL:
            # Mask keys beyond the query position.
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            qk = tl.where(k_pos <= q_pos, qk, -float("inf"))

        # Stable online softmax update.
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        if USE_DROPOUT:
            # Dropout on *softmax probabilities* (PyTorch semantics): apply mask/scale
            # to the numerator only; denominator (lse) remains unchanged.
            keep_prob = 1.0 - float(DROPOUT_P)
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            rng = tl.rand(seed + tl.full([], z, tl.uint32), q_pos * T + k_pos)
            keep = rng < keep_prob
            p_num = p * keep.to(tl.float32) * (1.0 / keep_prob)
        else:
            p_num = p

        acc = acc * alpha[:, None] + tl.dot(p_num, v)

        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + z * stride_oz + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        out,
        mask=m_mask[:, None] & d_mask[None, :],
    )
    lse = m_i + tl.log(l_i)
    tl.store(
        lse_ptr + z * stride_lsez + offs_m * stride_lset,
        lse,
        mask=m_mask,
    )
