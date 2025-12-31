"""Tests for the MOSAIC block layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from caramba.config.layer import LayerType, MosaicBlockLayerConfig


class MosaicBlockLayerTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=32,
            conv_kernel=5,
            mlp_mult=2.0,
            dropout_p=0.0,
            state_k=4,
            mem_buckets=256,
            mem_dim=16,
            mem_hashes=2,
            mem_write_threshold=0.99,  # mostly off
            mem_write_eta=0.1,
        )
        layer = cfg.build()
        x = torch.randn(2, 7, 32)
        y = layer(x, ctx=None)  # type: ignore[call-arg]
        self.assertEqual(tuple(y.shape), (2, 7, 32))

    def test_streaming_matches_full(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=16,
            conv_kernel=7,
            mlp_mult=1.5,
            dropout_p=0.0,
            state_k=3,
            state_decay_min=0.9,
            state_decay_max=0.99,
            mem_buckets=128,
            mem_dim=8,
            mem_hashes=2,
            mem_write_threshold=0.5,
            mem_write_eta=0.2,
        )
        layer = cfg.build()
        layer.eval()

        B, T, D = 1, 12, 16
        x = torch.randn(B, T, D)

        # Full sequence.
        y_full = layer(x, ctx=None)  # type: ignore[call-arg]

        # Streaming one token at a time with persistent ctx.
        ctx = SimpleNamespace()
        ys = []
        for t in range(T):
            yt = layer(x[:, t : t + 1, :], ctx=ctx)  # type: ignore[call-arg]
            ys.append(yt)
        y_stream = torch.cat(ys, dim=1)

        self.assertTrue(torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

