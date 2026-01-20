from __future__ import annotations

import torch

from layer.ssm import _SelectiveScan


class TestSelectiveScan:
    @staticmethod
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

    def test_matches_reference_forward_cpu(self) -> None:
        torch.manual_seed(0)
        Bsz, T, D_inner, D_state = 2, 64, 32, 8
        x = torch.randn(Bsz, T, D_inner, dtype=torch.float32)
        dt = torch.rand(Bsz, T, D_inner, dtype=torch.float32)
        A = -torch.exp(torch.randn(D_inner, D_state, dtype=torch.float32))
        B = torch.randn(Bsz, T, D_state, dtype=torch.float32)
        C = torch.randn(Bsz, T, D_state, dtype=torch.float32)
        D = torch.randn(D_inner, dtype=torch.float32)

        y_ref = self._reference_scan(x=x, dt=dt, A=A, B=B, C=C, D=D)
        y_new = _SelectiveScan.selective_scan(x=x, dt=dt, A=A, B=B, C=C, D=D)

        torch.testing.assert_close(y_new, y_ref, rtol=1e-4, atol=1e-5)

    def test_backward_agrees_with_reference_cpu(self) -> None:
        torch.manual_seed(1)
        Bsz, T, D_inner, D_state = 2, 32, 16, 4
        x0 = torch.randn(Bsz, T, D_inner, dtype=torch.float32, requires_grad=True)
        dt0 = torch.rand(Bsz, T, D_inner, dtype=torch.float32, requires_grad=True)
        A0 = (-torch.exp(torch.randn(D_inner, D_state, dtype=torch.float32))).requires_grad_()
        B0 = torch.randn(Bsz, T, D_state, dtype=torch.float32, requires_grad=True)
        C0 = torch.randn(Bsz, T, D_state, dtype=torch.float32, requires_grad=True)
        D0 = torch.randn(D_inner, dtype=torch.float32, requires_grad=True)

        y_ref = self._reference_scan(x=x0, dt=dt0, A=A0, B=B0, C=C0, D=D0)
        loss_ref = (y_ref ** 2).mean()
        loss_ref.backward()
        grads_ref = (x0.grad, dt0.grad, A0.grad, B0.grad, C0.grad, D0.grad)

        x1 = x0.detach().clone().requires_grad_()
        dt1 = dt0.detach().clone().requires_grad_()
        A1 = A0.detach().clone().requires_grad_()
        B1 = B0.detach().clone().requires_grad_()
        C1 = C0.detach().clone().requires_grad_()
        D1 = D0.detach().clone().requires_grad_()

        y_new = _SelectiveScan.selective_scan(x=x1, dt=dt1, A=A1, B=B1, C=C1, D=D1)
        loss_new = (y_new ** 2).mean()
        loss_new.backward()
        grads_new = (x1.grad, dt1.grad, A1.grad, B1.grad, C1.grad, D1.grad)

        for g_new, g_ref in zip(grads_new, grads_ref, strict=True):
            assert g_new is not None and g_ref is not None
            torch.testing.assert_close(g_new, g_ref, rtol=2e-4, atol=2e-5)

