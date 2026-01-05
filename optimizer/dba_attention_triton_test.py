from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from caramba.optimizer.runtime import TRITON_AVAILABLE


def _skip_if_no_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton is not available")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_dba_attention_forward_matches_sdpa(dtype: torch.dtype, causal: bool) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    device = torch.device("cuda")
    B, H, T = 2, 4, 65
    D_sem, D_geo, D_v = 32, 32, 64
    sem_scale = 1.0 / math.sqrt(float(D_sem))
    geo_scale = 1.0 / math.sqrt(float(D_geo))

    q_sem = torch.randn((B, H, T, D_sem), device=device, dtype=dtype).contiguous()
    q_geo = torch.randn((B, H, T, D_geo), device=device, dtype=dtype).contiguous()
    k_sem = torch.randn((B, H, T, D_sem), device=device, dtype=dtype).contiguous()
    k_geo = torch.randn((B, H, T, D_geo), device=device, dtype=dtype).contiguous()
    v = torch.randn((B, H, T, D_v), device=device, dtype=dtype).contiguous()

    from caramba.optimizer.dba_attention_triton import DecoupledAttentionTraining

    y = DecoupledAttentionTraining().run(
        q_sem=q_sem,
        q_geo=q_geo,
        k_sem=k_sem,
        k_geo=k_geo,
        v=v,
        causal=causal,
        sem_scale=sem_scale,
        geo_scale=geo_scale,
        dropout_p=0.0,
    )

    q_cat = torch.cat([q_sem * sem_scale, q_geo * geo_scale], dim=-1)
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    y_ref = F.scaled_dot_product_attention(q_cat, k_cat, v, attn_mask=None, dropout_p=0.0, is_causal=bool(causal), scale=1.0)
    torch.testing.assert_close(y, y_ref, rtol=6e-2, atol=6e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_dba_attention_backward_matches_sdpa(dtype: torch.dtype, causal: bool) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(1)

    device = torch.device("cuda")
    B, H, T = 2, 4, 33
    D_sem, D_geo, D_v = 32, 32, 64
    sem_scale = 1.0 / math.sqrt(float(D_sem))
    geo_scale = 1.0 / math.sqrt(float(D_geo))

    q_sem0 = torch.randn((B, H, T, D_sem), device=device, dtype=dtype, requires_grad=True).contiguous()
    q_geo0 = torch.randn((B, H, T, D_geo), device=device, dtype=dtype, requires_grad=True).contiguous()
    k_sem0 = torch.randn((B, H, T, D_sem), device=device, dtype=dtype, requires_grad=True).contiguous()
    k_geo0 = torch.randn((B, H, T, D_geo), device=device, dtype=dtype, requires_grad=True).contiguous()
    v0 = torch.randn((B, H, T, D_v), device=device, dtype=dtype, requires_grad=True).contiguous()

    from caramba.optimizer.dba_attention_triton import DecoupledAttentionTraining

    y = DecoupledAttentionTraining().run(
        q_sem=q_sem0,
        q_geo=q_geo0,
        k_sem=k_sem0,
        k_geo=k_geo0,
        v=v0,
        causal=causal,
        sem_scale=sem_scale,
        geo_scale=geo_scale,
        dropout_p=0.0,
    )
    loss = (y.float() ** 2).mean()
    loss.backward()
    dq_sem = q_sem0.grad
    dq_geo = q_geo0.grad
    dk_sem = k_sem0.grad
    dk_geo = k_geo0.grad
    dv = v0.grad
    assert dq_sem is not None and dq_geo is not None and dk_sem is not None and dk_geo is not None and dv is not None

    q_sem1 = q_sem0.detach().clone().requires_grad_()
    q_geo1 = q_geo0.detach().clone().requires_grad_()
    k_sem1 = k_sem0.detach().clone().requires_grad_()
    k_geo1 = k_geo0.detach().clone().requires_grad_()
    v1 = v0.detach().clone().requires_grad_()
    q_cat = torch.cat([q_sem1 * sem_scale, q_geo1 * geo_scale], dim=-1)
    k_cat = torch.cat([k_sem1, k_geo1], dim=-1)
    y_ref = F.scaled_dot_product_attention(q_cat, k_cat, v1, attn_mask=None, dropout_p=0.0, is_causal=bool(causal), scale=1.0)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()

    dq_cat = q_cat.grad
    dk_cat = k_cat.grad
    assert dq_cat is not None and dk_cat is not None
    dq_sem_ref = dq_cat[..., :D_sem] * sem_scale
    dq_geo_ref = dq_cat[..., D_sem:] * geo_scale
    dk_sem_ref = dk_cat[..., :D_sem]
    dk_geo_ref = dk_cat[..., D_sem:]
    dv_ref = v1.grad
    assert dv_ref is not None

    torch.testing.assert_close(dq_sem, dq_sem_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dq_geo, dq_geo_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dk_sem, dk_sem_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dk_geo, dk_geo_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=8e-2, atol=8e-2)


def test_triton_dba_attention_dropout_is_deterministic_given_seed() -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(2)

    device = torch.device("cuda")
    dtype = torch.float16
    B, H, T = 1, 2, 32
    D_sem, D_geo, D_v = 32, 32, 64
    sem_scale = 1.0 / math.sqrt(float(D_sem))
    geo_scale = 1.0 / math.sqrt(float(D_geo))
    dropout_p = 0.25

    q_sem = torch.randn((B, H, T, D_sem), device=device, dtype=dtype).contiguous()
    q_geo = torch.randn((B, H, T, D_geo), device=device, dtype=dtype).contiguous()
    k_sem = torch.randn((B, H, T, D_sem), device=device, dtype=dtype).contiguous()
    k_geo = torch.randn((B, H, T, D_geo), device=device, dtype=dtype).contiguous()
    v = torch.randn((B, H, T, D_v), device=device, dtype=dtype).contiguous()

    from caramba.optimizer.dba_attention_triton import DecoupledAttentionTraining

    seed = 12345
    y0 = DecoupledAttentionTraining().run(
        q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, v=v,
        causal=True, sem_scale=sem_scale, geo_scale=geo_scale,
        dropout_p=dropout_p, seed=seed,
    )
    y1 = DecoupledAttentionTraining().run(
        q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, v=v,
        causal=True, sem_scale=sem_scale, geo_scale=geo_scale,
        dropout_p=dropout_p, seed=seed,
    )
    torch.testing.assert_close(y0, y1, rtol=0.0, atol=0.0)
    y2 = DecoupledAttentionTraining().run(
        q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, v=v,
        causal=True, sem_scale=sem_scale, geo_scale=geo_scale,
        dropout_p=dropout_p, seed=seed + 1,
    )
    assert not torch.allclose(y0, y2)

