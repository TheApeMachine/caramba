from __future__ import annotations

import pytest
import torch

from caramba.optimizer.runtime import TRITON_AVAILABLE


def _skip_if_no_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton is not available")


def _rope_ref(*, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: int) -> torch.Tensor:
    T = int(x.shape[2])
    cos2 = cos[:T].unsqueeze(0).unsqueeze(0)
    sin2 = sin[:T].unsqueeze(0).unsqueeze(0)
    x_rot = x[..., :rot]
    x_pass = x[..., rot:]
    x1 = x_rot[..., : rot // 2]
    x2 = x_rot[..., rot // 2 : rot]
    y1 = x1 * cos2 - x2 * sin2
    y2 = x1 * sin2 + x2 * cos2
    return torch.cat([y1, y2, x_pass], dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rope_forward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(0)

    device = torch.device("cuda")
    B, H, T, D = 2, 4, 33, 128
    rot = 64

    x = torch.randn((B, H, T, D), device=device, dtype=dtype).contiguous()
    t = torch.arange(0, T, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, device=device, dtype=torch.float32) / float(rot)))
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype).contiguous()
    sin = torch.sin(freqs).to(dtype).contiguous()

    from caramba.optimizer.rope_triton import rope_triton

    y = rope_triton(x=x, cos=cos, sin=sin, rot_dim=rot)
    y_ref = _rope_ref(x=x.to(torch.float32), cos=cos.to(torch.float32), sin=sin.to(torch.float32), rot=rot).to(dtype)
    torch.testing.assert_close(y, y_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rope_backward_matches_reference(dtype: torch.dtype) -> None:
    _skip_if_no_cuda_triton()
    torch.manual_seed(1)

    device = torch.device("cuda")
    B, H, T, D = 2, 4, 17, 128
    rot = 64

    x0 = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True).contiguous()
    t = torch.arange(0, T, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, device=device, dtype=torch.float32) / float(rot)))
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype).contiguous()
    sin = torch.sin(freqs).to(dtype).contiguous()

    from caramba.optimizer.rope_triton import rope_triton

    y = rope_triton(x=x0, cos=cos, sin=sin, rot_dim=rot)
    loss = (y.float() ** 2).mean()
    loss.backward()
    gx = x0.grad
    assert gx is not None

    x1 = x0.detach().clone().requires_grad_()
    y_ref = _rope_ref(x=x1.to(torch.float32), cos=cos.to(torch.float32), sin=sin.to(torch.float32), rot=rot).to(dtype)
    loss_ref = (y_ref.float() ** 2).mean()
    loss_ref.backward()
    gx_ref = x1.grad
    assert gx_ref is not None

    torch.testing.assert_close(gx, gx_ref, rtol=8e-2, atol=8e-2)

