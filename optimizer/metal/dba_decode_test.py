from __future__ import annotations

import math

import pytest
import torch

from optimizer.runtime import METAL_SUPPORTED


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_dba_decode_matches_reference() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")

    # Build/load the extension; if the toolchain isn't present, skip rather than fail.
    try:
        from optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from optimizer.metal import dba_decode_fp16

    device = torch.device("mps")
    dtype = torch.float16

    # Use a larger preallocated KV buffer and slice it to validate stride handling.
    B, H = 2, 4
    sem_hd, geo_hd, v_hd = 16, 16, 32
    S = 385  # not a multiple of 256 → exercises the tail block
    maxS = 512

    sem_scale = 1.0 / math.sqrt(float(sem_hd))
    geo_scale = 1.0 / math.sqrt(float(geo_hd))

    q_sem = torch.randn((B, H, 1, sem_hd), device=device, dtype=dtype)
    q_geo = torch.randn((B, H, 1, geo_hd), device=device, dtype=dtype)

    k_sem_buf = torch.randn((B, maxS, H * sem_hd), device=device, dtype=dtype)
    k_geo_buf = torch.randn((B, maxS, H * geo_hd), device=device, dtype=dtype)
    v_buf = torch.randn((B, maxS, H * v_hd), device=device, dtype=dtype)

    k_sem = k_sem_buf.narrow(1, 0, S)
    k_geo = k_geo_buf.narrow(1, 0, S)
    v = v_buf.narrow(1, 0, S)

    out_metal = dba_decode_fp16(
        q_sem=q_sem,
        q_geo=q_geo,
        k_sem=k_sem,
        k_geo=k_geo,
        v=v,
        sem_scale=sem_scale,
        geo_scale=geo_scale,
    )

    # Reference (PyTorch) in fp32 for numerics.
    q_sem_f = q_sem.to(torch.float32)
    q_geo_f = q_geo.to(torch.float32)
    k_sem_h = k_sem.view(B, S, H, sem_hd).transpose(1, 2).contiguous().to(torch.float32)  # (B,H,S,sem_hd)
    k_geo_h = k_geo.view(B, S, H, geo_hd).transpose(1, 2).contiguous().to(torch.float32)
    v_h = v.view(B, S, H, v_hd).transpose(1, 2).contiguous().to(torch.float32)  # (B,H,S,v_hd)

    sem_scores = torch.matmul(q_sem_f, k_sem_h.transpose(-2, -1)) * float(sem_scale)  # (B,H,1,S)
    geo_scores = torch.matmul(q_geo_f, k_geo_h.transpose(-2, -1)) * float(geo_scale)
    scores = sem_scores + geo_scores
    attn = torch.softmax(scores, dim=-1)  # (B,H,1,S)
    out_ref = torch.matmul(attn, v_h).to(dtype)  # (B,H,1,v_hd)

    # MPS fp16 has some drift; keep tolerances reasonable.
    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_dba_decode_with_null_matches_reference() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")

    try:
        from optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from optimizer.metal import dba_decode_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, H = 2, 4
    sem_hd, geo_hd, v_hd = 16, 16, 32
    S = 385
    maxS = 512

    sem_scale = 1.0 / math.sqrt(float(sem_hd))
    geo_scale = 1.0 / math.sqrt(float(geo_hd))

    q_sem = torch.randn((B, H, 1, sem_hd), device=device, dtype=dtype)
    q_geo = torch.randn((B, H, 1, geo_hd), device=device, dtype=dtype)

    k_sem_buf = torch.randn((B, maxS, H * sem_hd), device=device, dtype=dtype)
    k_geo_buf = torch.randn((B, maxS, H * geo_hd), device=device, dtype=dtype)
    v_buf = torch.randn((B, maxS, H * v_hd), device=device, dtype=dtype)

    k_sem = k_sem_buf.narrow(1, 0, S)
    k_geo = k_geo_buf.narrow(1, 0, S)
    v = v_buf.narrow(1, 0, S)

    k_sem_null = torch.randn((B, H, 1, sem_hd), device=device, dtype=dtype)
    k_geo_null = torch.randn((B, H, 1, geo_hd), device=device, dtype=dtype)
    v_null = torch.randn((B, H, 1, v_hd), device=device, dtype=dtype)

    out_metal = dba_decode_fp16(
        q_sem=q_sem,
        q_geo=q_geo,
        k_sem=k_sem,
        k_geo=k_geo,
        v=v,
        k_sem_null=k_sem_null,
        k_geo_null=k_geo_null,
        v_null=v_null,
        sem_scale=sem_scale,
        geo_scale=geo_scale,
    )

    # Reference (PyTorch) with explicit null token.
    q_sem_f = q_sem.to(torch.float32)
    q_geo_f = q_geo.to(torch.float32)
    k_sem_h = k_sem.view(B, S, H, sem_hd).transpose(1, 2).contiguous().to(torch.float32)  # (B,H,S,sem_hd)
    k_geo_h = k_geo.view(B, S, H, geo_hd).transpose(1, 2).contiguous().to(torch.float32)
    v_h = v.view(B, S, H, v_hd).transpose(1, 2).contiguous().to(torch.float32)  # (B,H,S,v_hd)

    ksn = k_sem_null[:, :, 0, :].contiguous().to(torch.float32)  # (B,H,sem_hd)
    kgn = k_geo_null[:, :, 0, :].contiguous().to(torch.float32)
    vn = v_null[:, :, 0, :].contiguous().to(torch.float32)  # (B,H,v_hd)

    score_null = (q_sem_f[:, :, 0:1, :] * ksn.unsqueeze(2)).sum(dim=-1, keepdim=True) * float(sem_scale) + \
                 (q_geo_f[:, :, 0:1, :] * kgn.unsqueeze(2)).sum(dim=-1, keepdim=True) * float(geo_scale)  # (B,H,1,1)

    sem_scores = torch.matmul(q_sem_f, k_sem_h.transpose(-2, -1)) * float(sem_scale)  # (B,H,1,S)
    geo_scores = torch.matmul(q_geo_f, k_geo_h.transpose(-2, -1)) * float(geo_scale)
    scores = sem_scores + geo_scores
    scores2 = torch.cat([score_null, scores], dim=-1)  # (B,H,1,1+S)
    v2 = torch.cat([vn.unsqueeze(2), v_h], dim=2)  # (B,H,1+S,v_hd)
    attn = torch.softmax(scores2, dim=-1)  # (B,H,1,1+S)
    out_ref = torch.matmul(attn, v2).to(dtype)  # (B,H,1,v_hd)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)

