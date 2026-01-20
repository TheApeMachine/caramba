from __future__ import annotations

import torch


class TestMetalRMSNormAutograd:
    def test_rmsnorm_backward_matches_pytorch(self) -> None:
        if not torch.backends.mps.is_available():
            return

        torch.manual_seed(0)
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D, device="mps", dtype=torch.float16, requires_grad=True)
        w = torch.randn(D, device="mps", dtype=torch.float16, requires_grad=True)
        eps = 1e-6

        # Metal
        from optimizer.metal.rmsnorm import rmsnorm_fp16

        y_m = rmsnorm_fp16(x=x, weight=w, eps=eps, verbose_build=False)
        loss_m = (y_m.float() ** 2).mean()
        loss_m.backward()
        assert x.grad is not None
        assert w.grad is not None
        gx_m = x.grad.detach().clone()
        gw_m = w.grad.detach().clone()

        # Reset grads
        x.grad = None
        w.grad = None

        # Reference (PyTorch)
        x_f = x.float()
        inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        y_ref = (x_f * inv).to(dtype=torch.float16) * w
        loss_ref = (y_ref.float() ** 2).mean()
        loss_ref.backward()
        gx_r = x.grad
        gw_r = w.grad
        assert gx_r is not None and gw_r is not None

        torch.testing.assert_close(gx_m, gx_r, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(gw_m, gw_r, rtol=5e-3, atol=5e-3)

