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
def test_triton_rmsnorm_forward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    B, T, D = 2, 17, 512
    eps = 1e-6
    device = torch.device("cuda")

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype)

    from caramba.optimizer.rmsnorm_triton import rmsnorm_triton

    y = rmsnorm_triton(x=x, weight=w, eps=eps)
    x_f = x.to(torch.float32)
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (x_f * inv).to(dtype) * w

    torch.testing.assert_close(y, y_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rmsnorm_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(1)

    B, T, D = 2, 9, 256
    eps = 1e-6
    device = torch.device("cuda")

    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    w0 = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    from caramba.optimizer.rmsnorm_triton import rmsnorm_triton

    y = rmsnorm_triton(x=x0, weight=w0, eps=eps)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    gw = w0.grad
    assert gx is not None and gw is not None

    x1 = x0.detach().clone().requires_grad_()
    w1 = w0.detach().clone().requires_grad_()
    x_f = x1.to(torch.float32)
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (x_f * inv).to(dtype) * w1
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    gw_ref = w1.grad
    assert gx_ref is not None and gw_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(gw, gw_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rmsnorm_noweight_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(2)

    B, T, D = 2, 9, 256
    eps = 1e-6
    device = torch.device("cuda")

    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    from caramba.optimizer.rmsnorm_triton import rmsnorm_triton

    y = rmsnorm_triton(x=x0, weight=None, eps=eps)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    assert gx is not None

    x1 = x0.detach().clone().requires_grad_()
    x_f = x1.to(torch.float32)
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (x_f * inv).to(dtype)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    assert gx_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)

