from __future__ import annotations

import pytest
import torch

from caramba.optimizer.runtime import metal_supported


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_rmsnorm_matches_reference() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")

    # Build/load the extension; if the toolchain isn't present, skip rather than fail.
    try:
        from caramba.optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from caramba.optimizer.metal import rmsnorm_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, T, D = 4, 17, 256
    eps = 1e-6

    x = torch.randn((B, T, D), device=device, dtype=dtype)
    w = torch.randn((D,), device=device, dtype=dtype)

    out_metal = rmsnorm_fp16(x=x, weight=w, eps=eps)

    # Reference in fp32 for better numerics.
    x_f = x.to(torch.float32)
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    out_ref = (x_f * inv).to(dtype) * w

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_rmsnorm_noweight_matches_reference() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")

    try:
        from caramba.optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from caramba.optimizer.metal import rmsnorm_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, T, D = 2, 8, 512
    eps = 1e-6

    x = torch.randn((B, T, D), device=device, dtype=dtype)
    out_metal = rmsnorm_fp16(x=x, weight=None, eps=eps)

    x_f = x.to(torch.float32)
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    out_ref = (x_f * inv).to(dtype)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=2e-2, rtol=2e-2)

