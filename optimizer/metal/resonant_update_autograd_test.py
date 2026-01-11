from __future__ import annotations

import torch


class TestMetalResonantUpdateAutograd:
    """Resonant update kernel tests.

    Validates that Metal fused forward+backward matches the PyTorch reference.
    """

    def test_backward_matches_pytorch_reference(self) -> None:
        if not torch.backends.mps.is_available():
            return

        torch.manual_seed(0)
        BT, H, D = 8, 2, 32
        x = torch.randn(BT, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        y = torch.randn(BT, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        vr = torch.randn(BT, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        vi = torch.randn(BT, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        diag = torch.rand(H, D, device="mps", dtype=torch.float32)
        scale = 0.01
        damping = 0.02
        zero_diag = True

        from caramba.optimizer.metal.resonant_update import MetalResonantPhaseUpdate

        upd = MetalResonantPhaseUpdate()
        xo, yo = upd.forward(x=x, y=y, vr=vr, vi=vi, diag=diag, scale=scale, damping=damping, zero_diag=zero_diag)
        loss_m = (xo.pow(2) + yo.pow(2)).mean()
        loss_m.backward()
        assert x.grad is not None and y.grad is not None and vr.grad is not None and vi.grad is not None
        gx_m = x.grad.detach().clone()
        gy_m = y.grad.detach().clone()
        gvr_m = vr.grad.detach().clone()
        gvi_m = vi.grad.detach().clone()

        x.grad = None
        y.grad = None
        vr.grad = None
        vi.grad = None

        inv_D = 1.0 / float(D)
        cr = vr * inv_D - diag.unsqueeze(0) * x
        ci = vi * inv_D - diag.unsqueeze(0) * y
        a = x * (1.0 - damping) + scale * cr
        b = y * (1.0 - damping) + scale * ci
        inv_r = torch.rsqrt(a * a + b * b + 1e-12)
        xo_r = a * inv_r
        yo_r = b * inv_r
        loss_r = (xo_r.pow(2) + yo_r.pow(2)).mean()
        loss_r.backward()

        torch.testing.assert_close(gx_m, x.grad, rtol=5e-4, atol=5e-4)
        torch.testing.assert_close(gy_m, y.grad, rtol=5e-4, atol=5e-4)
        torch.testing.assert_close(gvr_m, vr.grad, rtol=5e-4, atol=5e-4)
        torch.testing.assert_close(gvi_m, vi.grad, rtol=5e-4, atol=5e-4)

