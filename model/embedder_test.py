"""
embedder_test provides tests to validate the
embedder module.
"""
from __future__ import annotations

import unittest

import torch
from model.embedder import Embedder
from config.embedder import (
    EmbedderType,
    NoEmbedderConfig,
    TokenEmbedderConfig,
)


class EmbedderTest(unittest.TestCase):
    """
    EmbedderTest provides tests to validate the
    embedder module.
    """
    def test_forward(self) -> None:
        """
        test the forward pass of the embedder.
        """
        embedder = Embedder(
            TokenEmbedderConfig(
                type=EmbedderType.TOKEN,
                vocab_size=128,
                d_model=128,
            )
        )
        x: torch.Tensor = torch.randint(0, 128, (1, 10))
        self.assertEqual(embedder(x).shape, (1, 10, 128))

    def test_forward_none(self) -> None:
        """
        test the forward pass with no embedder.
        """
        embedder = Embedder(NoEmbedderConfig(type=EmbedderType.NONE))
        x: torch.Tensor = torch.randn(1, 10, 16)
        self.assertIs(embedder(x), x)

    def test_token_ids_not_corrupted_by_fp16_weights(self) -> None:
        """
        Token IDs must remain integer indices even if embedding weights are fp16.

        Regression test: casting token IDs to fp16 and back to long silently
        corrupts IDs above ~2048 (float16 mantissa limit), breaking Llama parity.
        """
        torch.manual_seed(0)
        embedder = Embedder(
            TokenEmbedderConfig(
                type=EmbedderType.TOKEN,
                vocab_size=10000,
                d_model=8,
            )
        ).to(dtype=torch.float16)

        # Include IDs above the fp16 exact-integer range.
        x = torch.tensor([[0, 1, 2047, 2048, 4097, 5000, 9999]], dtype=torch.long)
        y = embedder(x)
        expected = embedder.token_embedding(x)  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(y, expected))
