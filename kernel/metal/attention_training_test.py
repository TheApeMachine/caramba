from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from caramba.optimizer.runtime import metal_supported


def _skip_if_no_metal_attention_extension() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")
    try:
        from caramba.optimizer.metal.attention_jit import load_caramba_metal_attention_ops

        _ = load_caramba_metal_attention_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal attention extension unavailable: {e}")


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
@pytest.mark.parametrize("causal", [False, True])
def test_metal_attention_training_forward_matches_sdpa(causal: bool) -> None:
    _skip_if_no_metal_attention_extension()
    torch.manual_seed(0)

    device = torch.device("mps")
    dtype = torch.float16
    B, H, T, D = 2, 4, 33, 64
    scale = 1.0 / math.sqrt(float(D))

    q = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    k = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    v = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()

    from caramba.optimizer.metal.attention_training import MetalAttentionTraining

    y = MetalAttentionTraining().run(q=q, k=k, v=v, causal=causal, scale=scale, dropout_p=0.0)
    y_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=bool(causal), scale=scale)
    torch.testing.assert_close(y, y_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
@pytest.mark.parametrize("causal", [False, True])
def test_metal_attention_training_backward_matches_sdpa(causal: bool) -> None:
    _skip_if_no_metal_attention_extension()
    torch.manual_seed(1)

    device = torch.device("mps")
    dtype = torch.float16
    B, H, T, D = 1, 4, 17, 64
    scale = 1.0 / math.sqrt(float(D))

    q0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    k0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    v0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()

    from caramba.optimizer.metal.attention_training import MetalAttentionTraining

    y = MetalAttentionTraining().run(q=q0, k=k0, v=v0, causal=causal, scale=scale, dropout_p=0.0)
    loss = (y.float() ** 2).mean()
    loss.backward()
    dq = q0.grad
    dk = k0.grad
    dv = v0.grad
    assert dq is not None and dk is not None and dv is not None

    q1 = q0.detach().clone().requires_grad_()
    k1 = k0.detach().clone().requires_grad_()
    v1 = v0.detach().clone().requires_grad_()
    y_ref = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=None, dropout_p=0.0, is_causal=bool(causal), scale=scale)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    dq_ref = q1.grad
    dk_ref = k1.grad
    dv_ref = v1.grad
    assert dq_ref is not None and dk_ref is not None and dv_ref is not None

    torch.testing.assert_close(dq, dq_ref, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(dk, dk_ref, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(dv, dv_ref, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_attention_training_dropout_is_deterministic_given_seed() -> None:
    _skip_if_no_metal_attention_extension()
    torch.manual_seed(2)

    device = torch.device("mps")
    dtype = torch.float16
    B, H, T, D = 1, 2, 16, 64
    scale = 1.0 / math.sqrt(float(D))
    dropout_p = 0.25

    q = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    k = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    v = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()

    from caramba.optimizer.metal.attention_training import MetalAttentionTraining

    seed = 12345
    y0 = MetalAttentionTraining().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed)
    y1 = MetalAttentionTraining().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed)
    torch.testing.assert_close(y0, y1, rtol=0.0, atol=0.0)

    y2 = MetalAttentionTraining().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed + 1)
    assert not torch.allclose(y0, y2)

