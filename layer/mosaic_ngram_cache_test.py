"""Tests for the MOSAIC n-gram cache logits layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from caramba.config.layer import LayerType, MosaicNGramCacheLogitsLayerConfig


class MosaicNGramCacheLogitsLayerTest(unittest.TestCase):
    def test_noop_without_ctx(self) -> None:
        cfg = MosaicNGramCacheLogitsLayerConfig(
            type=LayerType.MOSAIC_NGRAM_CACHE,
            vocab_size=50,
            n=4,
            table_size=64,
            top_m=8,
            weight=1.0,
        )
        layer = cfg.build()
        logits = torch.zeros(2, 5, 50)
        out = layer(logits, ctx=None)  # type: ignore[call-arg]
        self.assertEqual(tuple(out.shape), (2, 5, 50))

    def test_runs_with_input_ids(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicNGramCacheLogitsLayerConfig(
            type=LayerType.MOSAIC_NGRAM_CACHE,
            vocab_size=32,
            n=3,
            table_size=64,
            top_m=4,
            weight=0.5,
        )
        layer = cfg.build()
        B, T, V = 1, 10, 32
        input_ids = torch.randint(0, V, (B, T))
        logits = torch.zeros(B, T, V)
        ctx = SimpleNamespace(input_ids=input_ids)
        out = layer(logits, ctx=ctx)  # type: ignore[call-arg]
        self.assertEqual(tuple(out.shape), (B, T, V))


if __name__ == "__main__":
    unittest.main()

