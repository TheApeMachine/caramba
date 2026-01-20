from __future__ import annotations

import pytest
import torch

from optimizer.runtime import metal_supported


def _skip_if_no_metal_extension() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")
    try:
        from optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_layernorm_backward_matches_pytorch() -> None:
    _skip_if_no_metal_extension()

    torch.manual_seed(0)
    B, T, D = 2, 8, 128
    eps = 1e-5

    x = torch.randn(B, T, D, device="mps", dtype=torch.float16, requires_grad=True)
    w = torch.randn(D, device="mps", dtype=torch.float16, requires_grad=True)
    b = torch.randn(D, device="mps", dtype=torch.float16, requires_grad=True)

    from optimizer.metal.layernorm import layernorm_fp16

    y_m = layernorm_fp16(x=x, weight=w, bias=b, eps=eps, verbose_build=False)
    loss_m = (y_m.float() ** 2).mean()
    loss_m.backward()

    assert x.grad is not None and w.grad is not None and b.grad is not None
    gx_m = x.grad.detach().clone()
    gw_m = w.grad.detach().clone()
    gb_m = b.grad.detach().clone()

    x.grad = None
    w.grad = None
    b.grad = None

    y_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=w.to(torch.float32),
        bias=b.to(torch.float32),
        eps=eps,
    ).to(torch.float16)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()

    assert x.grad is not None and w.grad is not None and b.grad is not None
    torch.testing.assert_close(gx_m, x.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gw_m, w.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gb_m, b.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_layernorm_weight_only_backward_matches_pytorch() -> None:
    _skip_if_no_metal_extension()

    torch.manual_seed(0)
    B, T, D = 2, 8, 128
    eps = 1e-5

    x = torch.randn(B, T, D, device="mps", dtype=torch.float16, requires_grad=True)
    w = torch.randn(D, device="mps", dtype=torch.float16, requires_grad=True)

    from optimizer.metal.layernorm import layernorm_fp16

    y_m = layernorm_fp16(x=x, weight=w, bias=None, eps=eps, verbose_build=False)
    loss_m = (y_m.float() ** 2).mean()
    loss_m.backward()

    assert x.grad is not None and w.grad is not None
    gx_m = x.grad.detach().clone()
    gw_m = w.grad.detach().clone()

    x.grad = None
    w.grad = None

    y_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=w.to(torch.float32),
        bias=None,
        eps=eps,
    ).to(torch.float16)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()

    assert x.grad is not None and w.grad is not None
    torch.testing.assert_close(gx_m, x.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gw_m, w.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_layernorm_noweight_backward_matches_pytorch() -> None:
    _skip_if_no_metal_extension()

    torch.manual_seed(0)
    B, T, D = 2, 8, 128
    eps = 1e-5

    x = torch.randn(B, T, D, device="mps", dtype=torch.float16, requires_grad=True)

    from optimizer.metal.layernorm import layernorm_fp16

    y_m = layernorm_fp16(x=x, weight=None, bias=None, eps=eps, verbose_build=False)
    loss_m = (y_m.float() ** 2).mean()
    loss_m.backward()

    assert x.grad is not None
    gx_m = x.grad.detach().clone()

    x.grad = None

    y_ref = torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape=(D,),
        weight=None,
        bias=None,
        eps=eps,
    ).to(torch.float16)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()

    assert x.grad is not None
    torch.testing.assert_close(gx_m, x.grad, rtol=1e-2, atol=1e-2)

