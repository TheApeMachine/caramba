from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from optimizer.runtime import triton_supported


def _skip_if_no_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not triton_supported():
        pytest.skip("Triton is not available")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_flash_attention_forward_matches_sdpa(dtype: torch.dtype, causal: bool) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    device = torch.device("cuda")
    B, H, T, D = 2, 4, 65, 64
    scale = 1.0 / math.sqrt(float(D))

    q = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    k = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    v = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()

    from optimizer.flash_attention_triton import FlashAttention

    y = FlashAttention().run(q=q, k=k, v=v, causal=causal, scale=scale, dropout_p=0.0)
    y_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=bool(causal), scale=scale)
    torch.testing.assert_close(y, y_ref, rtol=6e-2, atol=6e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_flash_attention_backward_matches_sdpa(dtype: torch.dtype, causal: bool) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(1)

    device = torch.device("cuda")
    B, H, T, D = 2, 4, 33, 64
    scale = 1.0 / math.sqrt(float(D))

    q0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    k0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    v0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()

    from optimizer.flash_attention_triton import FlashAttention

    y = FlashAttention().run(q=q0, k=k0, v=v0, causal=causal, scale=scale, dropout_p=0.0)
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

    torch.testing.assert_close(dq, dq_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dk, dk_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=8e-2, atol=8e-2)


def test_triton_flash_attention_dropout_is_deterministic_given_seed() -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(2)

    device = torch.device("cuda")
    dtype = torch.float16
    B, H, T, D = 1, 2, 32, 64
    scale = 1.0 / math.sqrt(float(D))
    dropout_p = 0.25

    q = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    k = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    v = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()

    from optimizer.flash_attention_triton import FlashAttention

    seed = 12345
    y0 = FlashAttention().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed)
    y1 = FlashAttention().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed)
    torch.testing.assert_close(y0, y1, rtol=0.0, atol=0.0)

    y2 = FlashAttention().run(q=q, k=k, v=v, causal=True, scale=scale, dropout_p=dropout_p, seed=seed + 1)
    assert not torch.allclose(y0, y2)

