"""
forward_invariants_test validates key invariants of DecoupledAttentionLayer forward.

This catches regressions that lead to NaNs/Infs or obvious shape contract breaks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

# Allow running under an installed `caramba` package or directly from a repo checkout.
try:
    from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType  # type: ignore[import-not-found]
    from caramba.layer.attention.decoupled.layer import DecoupledAttentionLayer  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # Repo root is the directory that contains `config/`, `layer/`, etc.
    # This file lives at: <repo_root>/layer/attention/decoupled/forward_invariants_test.py
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from config.layer import AttentionLayerConfig, AttentionMode
    from layer.attention.decoupled.layer import DecoupledAttentionLayer


class DecoupledForwardInvariantsTest(unittest.TestCase):
    def _make_cfg(self, *, null_attn: bool, gate: bool, rope_enabled: bool) -> AttentionLayerConfig:
        # Keep QK smaller than V to exercise pad-to-v_head_dim logic.
        return AttentionLayerConfig(
            # Config discriminator uses canonical layer type names (see LayerType enum).
            type=LayerType.ATTENTION,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            mode=AttentionMode.DECOUPLED,
            attn_dim=32,  # V dim
            sem_dim=8,
            geo_dim=8,
            rope_enabled=bool(rope_enabled),
            rope_base=10000.0,
            is_causal=True,
            dropout_p=0.0,
            bias=False,
            null_attn=bool(null_attn),
            decoupled_gate=bool(gate),
        )

    def test_forward_is_finite_basic(self) -> None:
        torch.manual_seed(0)
        layer = DecoupledAttentionLayer(self._make_cfg(null_attn=False, gate=True, rope_enabled=False)).eval()
        x = torch.randn(2, 8, 32)
        y, cache = layer(x, ctx=None)
        self.assertIsNone(cache)
        self.assertEqual(tuple(y.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_is_finite_with_null_attn(self) -> None:
        torch.manual_seed(0)
        layer = DecoupledAttentionLayer(self._make_cfg(null_attn=True, gate=True, rope_enabled=False)).eval()
        x = torch.randn(2, 8, 32)
        y, cache = layer(x, ctx=None)
        self.assertIsNone(cache)
        self.assertEqual(tuple(y.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_is_finite_with_rope(self) -> None:
        torch.manual_seed(0)
        layer = DecoupledAttentionLayer(self._make_cfg(null_attn=False, gate=False, rope_enabled=True)).eval()
        x = torch.randn(2, 8, 32)
        y, cache = layer(x, ctx=None)
        self.assertIsNone(cache)
        self.assertEqual(tuple(y.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()

