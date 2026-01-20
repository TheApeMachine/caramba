from __future__ import annotations

import pytest
import torch

from optimizer.runtime import triton_supported


def _skip_if_no_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not triton_supported():
        pytest.skip("Triton is not available")


def _adamw_master_step_ref(
    *,
    p: torch.Tensor,
    g: torch.Tensor,
    master: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_size: float,
    beta1: float,
    beta2: float,
    eps: float,
    lr_wd: float,
) -> None:
    # Matches Metal kernel semantics exactly.
    master.mul_(1.0 - lr_wd)
    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)
    master.addcdiv_(exp_avg, denom, value=-step_size)
    p.copy_(master.to(dtype=p.dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_adamw_master_step_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    device = torch.device("cuda")
    n = 8192

    p = torch.randn((n,), device=device, dtype=dtype).contiguous()
    g = torch.randn((n,), device=device, dtype=dtype).contiguous()
    master = p.float().clone().contiguous()
    exp_avg = torch.randn((n,), device=device, dtype=torch.float32).contiguous()
    exp_avg_sq = torch.rand((n,), device=device, dtype=torch.float32).contiguous()

    p_ref = p.clone()
    master_ref = master.clone()
    m_ref = exp_avg.clone()
    v_ref = exp_avg_sq.clone()

    step_size = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_wd = 1e-4

    from optimizer.adamw_triton import adamw_triton_master_step

    adamw_triton_master_step(
        p=p,
        grad=g,
        master=master,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        lr_wd=lr_wd,
    )

    _adamw_master_step_ref(
        p=p_ref,
        g=g.float(),
        master=master_ref,
        exp_avg=m_ref,
        exp_avg_sq=v_ref,
        step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        lr_wd=lr_wd,
    )

    torch.testing.assert_close(master, master_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(exp_avg, m_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(exp_avg_sq, v_ref, rtol=1e-5, atol=1e-5)
    # Param is cast back to dtype; allow bf16 tolerance.
    torch.testing.assert_close(p.float(), p_ref.float(), rtol=2e-3, atol=2e-3)

