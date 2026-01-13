from __future__ import annotations

import pytest
import torch

from caramba.optimizer.runtime import triton_supported


def _skip_if_no_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not triton_supported():
        pytest.skip("Triton is not available")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_layernorm_forward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    B, T, D = 2, 11, 512
    eps = 1e-5
    device = torch.device("cuda")

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype)
    b = torch.randn(D, device=device, dtype=dtype)

    from caramba.optimizer.layernorm_triton import layernorm_triton

    y = layernorm_triton(x=x, weight=w, bias=b, eps=eps)
    y_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=w.to(torch.float32),
        bias=b.to(torch.float32),
        eps=eps,
    ).to(dtype)
    torch.testing.assert_close(y, y_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_layernorm_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(1)

    B, T, D = 2, 9, 256
    eps = 1e-5
    device = torch.device("cuda")

    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    w0 = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    b0 = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    from caramba.optimizer.layernorm_triton import layernorm_triton

    y = layernorm_triton(x=x0, weight=w0, bias=b0, eps=eps)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    gw = w0.grad
    gb = b0.grad
    assert gx is not None and gw is not None and gb is not None

    x1 = x0.detach().clone().requires_grad_()
    w1 = w0.detach().clone().requires_grad_()
    b1 = b0.detach().clone().requires_grad_()
    y_ref = torch.nn.functional.layer_norm(
        x1.to(torch.float32),
        normalized_shape=(D,),
        weight=w1.to(torch.float32),
        bias=b1.to(torch.float32),
        eps=eps,
    ).to(dtype)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    gw_ref = w1.grad
    gb_ref = b1.grad
    assert gx_ref is not None and gw_ref is not None and gb_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(gw, gw_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(gb, gb_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_layernorm_weight_only_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(2)

    B, T, D = 2, 9, 256
    eps = 1e-5
    device = torch.device("cuda")

    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    w0 = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    from caramba.optimizer.layernorm_triton import layernorm_triton

    y = layernorm_triton(x=x0, weight=w0, bias=None, eps=eps)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    gw = w0.grad
    assert gx is not None and gw is not None

    x1 = x0.detach().clone().requires_grad_()
    w1 = w0.detach().clone().requires_grad_()
    y_ref = torch.nn.functional.layer_norm(
        x1.to(torch.float32),
        normalized_shape=(D,),
        weight=w1.to(torch.float32),
        bias=None,
        eps=eps,
    ).to(dtype)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    gw_ref = w1.grad
    assert gx_ref is not None and gw_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(gw, gw_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_layernorm_noweight_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(3)

    B, T, D = 2, 9, 256
    eps = 1e-5
    device = torch.device("cuda")

    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    from caramba.optimizer.layernorm_triton import layernorm_triton

    y = layernorm_triton(x=x0, weight=None, bias=None, eps=eps)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    assert gx is not None

    x1 = x0.detach().clone().requires_grad_()
    y_ref = torch.nn.functional.layer_norm(
        x1.to(torch.float32),
        normalized_shape=(D,),
        weight=None,
        bias=None,
        eps=eps,
    ).to(dtype)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    assert gx_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)

