from __future__ import annotations

import pytest
import torch

from caramba.optimizer.runtime import METAL_SUPPORTED


@pytest.fixture(scope="session")
def metal_ops() -> object:
    """Build/load the Metal extension or skip the suite."""
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")
    try:
        from caramba.optimizer.metal.jit import load_caramba_metal_ops

        return load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_layernorm_matches_reference(metal_ops: object) -> None:
    _ = metal_ops
    from caramba.optimizer.metal import layernorm_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, T, D = 3, 19, 384
    eps = 1e-5

    x = torch.randn((B, T, D), device=device, dtype=dtype)
    w = torch.randn((D,), device=device, dtype=dtype)
    b = torch.randn((D,), device=device, dtype=dtype)

    out_metal = layernorm_fp16(x=x, weight=w, bias=b, eps=eps)

    # Reference in fp32 for numerics.
    out_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=w.to(torch.float32),
        bias=b.to(torch.float32),
        eps=eps,
    ).to(dtype)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_layernorm_weight_matches_reference(metal_ops: object) -> None:
    _ = metal_ops
    from caramba.optimizer.metal import layernorm_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, T, D = 2, 7, 256
    eps = 1e-5

    x = torch.randn((B, T, D), device=device, dtype=dtype)
    w = torch.randn((D,), device=device, dtype=dtype)

    out_metal = layernorm_fp16(x=x, weight=w, bias=None, eps=eps)

    out_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=w.to(torch.float32),
        bias=None,
        eps=eps,
    ).to(dtype)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_layernorm_noweight_matches_reference(metal_ops: object) -> None:
    _ = metal_ops
    from caramba.optimizer.metal import layernorm_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, D = 8, 128
    eps = 1e-5

    x = torch.randn((B, D), device=device, dtype=dtype)
    out_metal = layernorm_fp16(x=x, weight=None, bias=None, eps=eps)

    out_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=None,
        bias=None,
        eps=eps,
    ).to(dtype)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)

