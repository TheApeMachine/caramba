"""Triton CUDA kernels for DBA (decoupled) full-sequence attention backward.

Backprop uses the FlashAttention identity with LSE:
  p = exp(logits - lse)
  ds = p * (dp - delta)

Where:
  logits = sem_scale * (q_sem @ k_sem^T) + geo_scale * (q_geo @ k_geo^T)
  dp = dO @ V^T
  delta = sum_d(out * dO) per query token

Gradients:
  dV     = P^T @ dO
  dQ_sem = (ds @ K_sem) * sem_scale
  dK_sem = (ds^T @ Q_sem) * sem_scale
  dQ_geo = (ds @ K_geo) * geo_scale
  dK_geo = (ds^T @ Q_geo) * geo_scale
"""

import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

__all__ = ["dba_attn_bwd_preprocess", "dba_attn_bwd_dkv", "dba_attn_bwd_dq"]


@triton.jit
def dba_attn_bwd_preprocess(
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
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_DV: tl.constexpr = 256,
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    z = pid_z
    m_start = pid_m * BLOCK_M

    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    dv = tl.arange(0, BLOCK_DV)
    mv = dv < D_V

    o = tl.load(
        out_ptr + z * stride_oz + offs_m[:, None] * stride_ot + dv[None, :] * stride_od,
        mask=m_mask[:, None] & mv[None, :],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + dv[None, :] * stride_dod,
        mask=m_mask[:, None] & mv[None, :],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + z * stride_deltaz + offs_m * stride_deltat, delta, mask=m_mask)


@triton.jit
def dba_attn_bwd_dkv(
    q_sem_ptr,
    q_geo_ptr,
    k_sem_ptr,
    k_geo_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_sem_ptr,
    dk_geo_ptr,
    dv_ptr,
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
    stride_doz: tl.constexpr,
    stride_dot: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_lsez: tl.constexpr,
    stride_lset: tl.constexpr,
    stride_deltaz: tl.constexpr,
    stride_deltat: tl.constexpr,
    stride_dksz: tl.constexpr,
    stride_dkst: tl.constexpr,
    stride_dksd: tl.constexpr,
    stride_dkgz: tl.constexpr,
    stride_dkgt: tl.constexpr,
    stride_dkgd: tl.constexpr,
    stride_dvz: tl.constexpr,
    stride_dvt: tl.constexpr,
    stride_dvd: tl.constexpr,
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
    pid_z = tl.program_id(0)
    pid_n = tl.program_id(1)
    z = pid_z
    n_start = pid_n * BLOCK_N

    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < T
    ds = tl.arange(0, BLOCK_DSEM)
    dg = tl.arange(0, BLOCK_DGEO)
    dv = tl.arange(0, BLOCK_DV)
    ms = ds < D_SEM
    mg = dg < D_GEO
    mv = dv < D_V

    k_sem = tl.load(
        k_sem_ptr + z * stride_ksz + offs_n[:, None] * stride_kst + ds[None, :] * stride_ksd,
        mask=n_mask[:, None] & ms[None, :],
        other=0.0,
    ).to(tl.float32)
    k_geo = tl.load(
        k_geo_ptr + z * stride_kgz + offs_n[:, None] * stride_kgt + dg[None, :] * stride_kgd,
        mask=n_mask[:, None] & mg[None, :],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + dv[None, :] * stride_vd,
        mask=n_mask[:, None] & mv[None, :],
        other=0.0,
    ).to(tl.float32)

    dk_sem = tl.zeros((BLOCK_N, BLOCK_DSEM), tl.float32)
    dk_geo = tl.zeros((BLOCK_N, BLOCK_DGEO), tl.float32)
    dV = tl.zeros((BLOCK_N, BLOCK_DV), tl.float32)

    for m_start in range(0, T, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < T

        q_sem = tl.load(
            q_sem_ptr + z * stride_qsz + offs_m[:, None] * stride_qst + ds[None, :] * stride_qsd,
            mask=m_mask[:, None] & ms[None, :],
            other=0.0,
        ).to(tl.float32)
        q_geo = tl.load(
            q_geo_ptr + z * stride_qgz + offs_m[:, None] * stride_qgt + dg[None, :] * stride_qgd,
            mask=m_mask[:, None] & mg[None, :],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + dv[None, :] * stride_dod,
            mask=m_mask[:, None] & mv[None, :],
            other=0.0,
        ).to(tl.float32)

        lse = tl.load(lse_ptr + z * stride_lsez + offs_m * stride_lset, mask=m_mask, other=-float("inf")).to(tl.float32)
        delta = tl.load(delta_ptr + z * stride_deltaz + offs_m * stride_deltat, mask=m_mask, other=0.0).to(tl.float32)

        sem = tl.dot(q_sem, tl.trans(k_sem)) * SEM_SCALE
        geo = tl.dot(q_geo, tl.trans(k_geo)) * GEO_SCALE
        logits = sem + geo
        if CAUSAL:
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            logits = tl.where(k_pos <= q_pos, logits, -float("inf"))

        p = tl.exp(logits - lse[:, None])
        if USE_DROPOUT:
            keep_prob = 1.0 - float(DROPOUT_P)
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            rng = tl.rand(seed + tl.full([], z, tl.uint32), q_pos * T + k_pos)
            keep = rng < keep_prob
            w = p * keep.to(tl.float32) * (1.0 / keep_prob)
        else:
            w = p

        dV += tl.dot(tl.trans(w), do)

        dp = tl.dot(do, tl.trans(v))
        dscores = w * (dp - delta[:, None])
        dk_sem += tl.dot(tl.trans(dscores), q_sem) * SEM_SCALE
        dk_geo += tl.dot(tl.trans(dscores), q_geo) * GEO_SCALE

    tl.store(
        dk_sem_ptr + z * stride_dksz + offs_n[:, None] * stride_dkst + ds[None, :] * stride_dksd,
        dk_sem,
        mask=n_mask[:, None] & ms[None, :],
    )
    tl.store(
        dk_geo_ptr + z * stride_dkgz + offs_n[:, None] * stride_dkgt + dg[None, :] * stride_dkgd,
        dk_geo,
        mask=n_mask[:, None] & mg[None, :],
    )
    tl.store(
        dv_ptr + z * stride_dvz + offs_n[:, None] * stride_dvt + dv[None, :] * stride_dvd,
        dV,
        mask=n_mask[:, None] & mv[None, :],
    )


@triton.jit
def dba_attn_bwd_dq(
    q_sem_ptr,
    q_geo_ptr,
    k_sem_ptr,
    k_geo_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_sem_ptr,
    dq_geo_ptr,
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
    stride_doz: tl.constexpr,
    stride_dot: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_lsez: tl.constexpr,
    stride_lset: tl.constexpr,
    stride_deltaz: tl.constexpr,
    stride_deltat: tl.constexpr,
    stride_dqsz: tl.constexpr,
    stride_dqst: tl.constexpr,
    stride_dqsd: tl.constexpr,
    stride_dqgz: tl.constexpr,
    stride_dqgt: tl.constexpr,
    stride_dqgd: tl.constexpr,
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
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    z = pid_z
    m_start = pid_m * BLOCK_M

    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    ds = tl.arange(0, BLOCK_DSEM)
    dg = tl.arange(0, BLOCK_DGEO)
    dv = tl.arange(0, BLOCK_DV)
    ms = ds < D_SEM
    mg = dg < D_GEO
    mv = dv < D_V

    q_sem = tl.load(
        q_sem_ptr + z * stride_qsz + offs_m[:, None] * stride_qst + ds[None, :] * stride_qsd,
        mask=m_mask[:, None] & ms[None, :],
        other=0.0,
    ).to(tl.float32)
    q_geo = tl.load(
        q_geo_ptr + z * stride_qgz + offs_m[:, None] * stride_qgt + dg[None, :] * stride_qgd,
        mask=m_mask[:, None] & mg[None, :],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + z * stride_doz + offs_m[:, None] * stride_dot + dv[None, :] * stride_dod,
        mask=m_mask[:, None] & mv[None, :],
        other=0.0,
    ).to(tl.float32)

    lse = tl.load(lse_ptr + z * stride_lsez + offs_m * stride_lset, mask=m_mask, other=-float("inf")).to(tl.float32)
    delta = tl.load(delta_ptr + z * stride_deltaz + offs_m * stride_deltat, mask=m_mask, other=0.0).to(tl.float32)

    dq_sem = tl.zeros((BLOCK_M, BLOCK_DSEM), tl.float32)
    dq_geo = tl.zeros((BLOCK_M, BLOCK_DGEO), tl.float32)

    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_sem = tl.load(
            k_sem_ptr + z * stride_ksz + offs_n[:, None] * stride_kst + ds[None, :] * stride_ksd,
            mask=n_mask[:, None] & ms[None, :],
            other=0.0,
        ).to(tl.float32)
        k_geo = tl.load(
            k_geo_ptr + z * stride_kgz + offs_n[:, None] * stride_kgt + dg[None, :] * stride_kgd,
            mask=n_mask[:, None] & mg[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + z * stride_vz + offs_n[:, None] * stride_vt + dv[None, :] * stride_vd,
            mask=n_mask[:, None] & mv[None, :],
            other=0.0,
        ).to(tl.float32)

        sem = tl.dot(q_sem, tl.trans(k_sem)) * SEM_SCALE
        geo = tl.dot(q_geo, tl.trans(k_geo)) * GEO_SCALE
        logits = sem + geo
        if CAUSAL:
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            logits = tl.where(k_pos <= q_pos, logits, -float("inf"))

        p = tl.exp(logits - lse[:, None])
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
        dscores = w * (dp - delta[:, None])

        dq_sem += tl.dot(dscores, k_sem) * SEM_SCALE
        dq_geo += tl.dot(dscores, k_geo) * GEO_SCALE

    tl.store(
        dq_sem_ptr + z * stride_dqsz + offs_m[:, None] * stride_dqst + ds[None, :] * stride_dqsd,
        dq_sem,
        mask=m_mask[:, None] & ms[None, :],
    )
    tl.store(
        dq_geo_ptr + z * stride_dqgz + offs_m[:, None] * stride_dqgt + dg[None, :] * stride_dqgd,
        dq_geo,
        mask=m_mask[:, None] & mg[None, :],
    )

