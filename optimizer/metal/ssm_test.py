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


def _reference_scan(*, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("x must be (B,T,D_inner)")
    Bsz, T, D_inner = x.shape
    D_state = int(A.shape[1])

    a = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D_inner, D_state))
    u = (dt.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)

    h = torch.zeros((Bsz, D_inner, D_state), device=x.device, dtype=x.dtype)
    ys: list[torch.Tensor] = []
    for t in range(T):
        h = a[:, t] * h + u[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))
    y = torch.stack(ys, dim=1)
    return y + x * D.view(1, 1, -1)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_ssm_scan_forward_matches_reference() -> None:
    _skip_if_no_metal_extension()
    torch.manual_seed(0)

    Bsz, T, D_inner, D_state = 2, 64, 32, 8
    device = torch.device("mps")
    dtype = torch.float16

    x = torch.randn(Bsz, T, D_inner, device=device, dtype=dtype)
    dt = torch.rand(Bsz, T, D_inner, device=device, dtype=dtype)
    A = (-torch.exp(torch.randn(D_inner, D_state, device=device, dtype=dtype))).contiguous()
    B = torch.randn(Bsz, T, D_state, device=device, dtype=dtype)
    C = torch.randn(Bsz, T, D_state, device=device, dtype=dtype)
    D = torch.randn(D_inner, device=device, dtype=dtype)

    from optimizer.metal import MetalSSMSelectiveScan

    y_metal = MetalSSMSelectiveScan().run(x=x, dt=dt, A=A, B=B, C=C, D=D, verbose_build=False)

    # Reference in fp32 for numerics, cast back to fp16 like other Metal tests.
    y_ref = _reference_scan(
        x=x.to(torch.float32),
        dt=dt.to(torch.float32),
        A=A.to(torch.float32),
        B=B.to(torch.float32),
        C=C.to(torch.float32),
        D=D.to(torch.float32),
    ).to(dtype)

    torch.testing.assert_close(y_metal, y_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_ssm_scan_backward_matches_reference() -> None:
    _skip_if_no_metal_extension()
    torch.manual_seed(1)

    Bsz, T, D_inner, D_state = 2, 32, 16, 4
    device = torch.device("mps")
    dtype = torch.float16

    x0 = torch.randn(Bsz, T, D_inner, device=device, dtype=dtype, requires_grad=True)
    dt0 = torch.rand(Bsz, T, D_inner, device=device, dtype=dtype, requires_grad=True)
    A0 = (-torch.exp(torch.randn(D_inner, D_state, device=device, dtype=dtype))).requires_grad_()
    B0 = torch.randn(Bsz, T, D_state, device=device, dtype=dtype, requires_grad=True)
    C0 = torch.randn(Bsz, T, D_state, device=device, dtype=dtype, requires_grad=True)
    D0 = torch.randn(D_inner, device=device, dtype=dtype, requires_grad=True)

    from optimizer.metal import MetalSSMSelectiveScan

    y_m = MetalSSMSelectiveScan().run(x=x0, dt=dt0, A=A0, B=B0, C=C0, D=D0, verbose_build=False)
    loss_m = (y_m.float() ** 2).mean()
    loss_m.backward()
    grads_m = (x0.grad, dt0.grad, A0.grad, B0.grad, C0.grad, D0.grad)
    for g in grads_m:
        assert g is not None

    x1 = x0.detach().clone().requires_grad_()
    dt1 = dt0.detach().clone().requires_grad_()
    A1 = A0.detach().clone().requires_grad_()
    B1 = B0.detach().clone().requires_grad_()
    C1 = C0.detach().clone().requires_grad_()
    D1 = D0.detach().clone().requires_grad_()

    y_r = _reference_scan(
        x=x1.to(torch.float32),
        dt=dt1.to(torch.float32),
        A=A1.to(torch.float32),
        B=B1.to(torch.float32),
        C=C1.to(torch.float32),
        D=D1.to(torch.float32),
    ).to(dtype)
    loss_r = (y_r.float() ** 2).mean()
    loss_r.backward()
    grads_r = (x1.grad, dt1.grad, A1.grad, B1.grad, C1.grad, D1.grad)
    for g in grads_r:
        assert g is not None

    for g_m, g_r in zip(grads_m, grads_r, strict=True):
        torch.testing.assert_close(g_m, g_r, rtol=5e-2, atol=5e-2)

