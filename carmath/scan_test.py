from __future__ import annotations

import torch

from carmath.scan import leaky_integrator_scan


def test_leaky_integrator_scan_accepts_common_decay_shapes() -> None:
    torch.manual_seed(0)

    B, K, T, D = 2, 8, 5, 4
    inp = torch.randn(B, K, T, D, dtype=torch.float32)
    s0 = torch.randn(B, K, D, dtype=torch.float32)
    dec = torch.linspace(0.90, 0.99, K, dtype=torch.float32)

    d1 = dec  # (K,)
    d2 = dec.view(1, K, 1)  # (1,K,1) - used by MosaicBlock fast path
    d3 = dec.view(K, 1, 1)  # (K,1,1)

    s1, last1 = leaky_integrator_scan(inp, s0, d1)
    s2, last2 = leaky_integrator_scan(inp, s0, d2)
    s3, last3 = leaky_integrator_scan(inp, s0, d3)

    torch.testing.assert_close(s1, s2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(s1, s3, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(last1, last2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(last1, last3, rtol=1e-6, atol=1e-6)

