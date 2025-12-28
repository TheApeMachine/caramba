from __future__ import annotations

import pytest
import torch

from optimizer.runtime import METAL_SUPPORTED


@pytest.mark.skipif(not METAL_SUPPORTED, reason="Metal/MPS not supported on this platform")
def test_metal_rope_matches_reference() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is not available")

    # Build/load the extension; if the toolchain isn't present, skip rather than fail.
    try:
        from optimizer.metal.jit import load_caramba_metal_ops

        _ = load_caramba_metal_ops(verbose=False)
    except Exception as e:
        pytest.skip(f"caramba metal extension unavailable: {e}")

    from optimizer.metal import rope_fp16

    device = torch.device("mps")
    dtype = torch.float16

    B, H, T, D = 2, 4, 33, 64
    rot = 32
    base = 10000.0
    pos_offset = 7

    x = torch.randn((B, H, T, D), device=device, dtype=dtype)

    # Build cos/sin in fp16 for the kernel.
    t = torch.arange(pos_offset, pos_offset + T, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rot, 2, device=device, dtype=torch.float32) / float(rot))
    )
    freqs = torch.outer(t, inv_freq)  # (T, rot/2)
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)

    out_metal = rope_fp16(x=x, cos=cos, sin=sin, rot_dim=rot)

    # Reference in fp32 for numerics.
    x_f = x.to(torch.float32)
    cos_f = cos.to(torch.float32).unsqueeze(0).unsqueeze(0)
    sin_f = sin.to(torch.float32).unsqueeze(0).unsqueeze(0)
    x_rot = x_f[..., :rot]
    x_pass = x_f[..., rot:]
    x1 = x_rot[..., : rot // 2]
    x2 = x_rot[..., rot // 2 : rot]
    y1 = x1 * cos_f - x2 * sin_f
    y2 = x1 * sin_f + x2 * cos_f
    out_ref = torch.cat([y1, y2, x_pass], dim=-1).to(dtype)

    assert out_metal.shape == out_ref.shape
    assert torch.allclose(out_metal, out_ref, atol=3e-2, rtol=3e-2)

