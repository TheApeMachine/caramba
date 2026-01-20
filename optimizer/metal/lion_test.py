from __future__ import annotations

import pytest
import torch

from optimizer.runtime import metal_supported


@pytest.mark.skipif(not metal_supported(), reason="Metal/MPS not supported on this platform")
def test_metal_lion_step_matches_reference() -> None:
    # Build/load the extension; if the toolchain isn't present, skip rather than fail.
    try:
        from optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from optimizer.metal import lion_fp16

    device = torch.device("mps")
    dtype = torch.float16

    n = 4096
    lr = 1e-3
    beta1 = 0.9
    wd = 0.01

    p = torch.randn((n,), device=device, dtype=dtype)
    g = torch.randn((n,), device=device, dtype=dtype)
    m = torch.randn((n,), device=device, dtype=dtype)

    p_ref = p.to(torch.float32)
    g_ref = g.to(torch.float32)
    m_ref = m.to(torch.float32)

    m1 = beta1 * m_ref + (1.0 - beta1) * g_ref
    p1 = p_ref * (1.0 - lr * wd) - lr * torch.sign(m1)

    out = lion_fp16(p=p, grad=g, m=m, lr=lr, beta1=beta1, weight_decay=wd)
    assert out is p
    assert torch.allclose(p, p1.to(dtype), atol=2e-2, rtol=2e-2)
    assert torch.allclose(m, m1.to(dtype), atol=2e-2, rtol=2e-2)

