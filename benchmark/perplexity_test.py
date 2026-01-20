"""
Unit tests for the perplexity benchmark module.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from config.benchmark import PerplexityBenchmarkConfig


class DummyLMModel(nn.Module):
    """A simple language model for testing."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        h = self.embed(x)
        return self.head(h)


class TestPerplexityResult(unittest.TestCase):
    """Tests for PerplexityResult dataclass."""

    def test_result_fields(self) -> None:
        """Result stores all required fields."""
        result = PerplexityResult(
            model_name="test_model",
            perplexity=10.5,
            loss=2.35,
            num_tokens=1000,
            num_batches=10,
        )
        self.assertEqual(result.model_name, "test_model")
        self.assertAlmostEqual(result.perplexity, 10.5)
        self.assertAlmostEqual(result.loss, 2.35)
        self.assertEqual(result.num_tokens, 1000)
        self.assertEqual(result.num_batches, 10)


class TestPerplexityBenchmark(unittest.TestCase):
    """Tests for PerplexityBenchmark."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.vocab_size = 100
        self.block_size = 32
        self.batch_size = 2
        self.num_batches = 5

        # Create temporary dataset
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_data.npy"

        # Generate random tokens
        num_tokens = self.block_size * self.batch_size * self.num_batches * 2
        data = np.random.randint(0, self.vocab_size, size=num_tokens, dtype=np.uint16)
        np.save(self.dataset_path, data)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_returns_result(self) -> None:
        """Benchmark run returns a PerplexityResult."""
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsInstance(result, PerplexityResult)
        self.assertEqual(result.model_name, "test_model")

    def test_perplexity_is_positive(self) -> None:
        """Computed perplexity is a positive number."""
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertGreater(result.perplexity, 0)
        self.assertFalse(math.isinf(result.perplexity))
        self.assertFalse(math.isnan(result.perplexity))

    def test_loss_is_positive(self) -> None:
        """Computed loss is a positive number."""
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertGreater(result.loss, 0)

    def test_perplexity_loss_relationship(self) -> None:
        """Perplexity equals exp(loss)."""
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        expected_ppl = math.exp(result.loss)
        self.assertAlmostEqual(result.perplexity, expected_ppl, places=4)

    def test_num_batches_limit(self) -> None:
        """Benchmark respects num_batches limit."""
        limit = 2
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=limit,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertEqual(result.num_batches, limit)

    def test_num_tokens_matches_config(self) -> None:
        """Number of tokens matches batch_size * block_size * num_batches."""
        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=self.block_size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        model = DummyLMModel(vocab_size=self.vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        expected_tokens = self.batch_size * self.block_size * result.num_batches
        self.assertEqual(result.num_tokens, expected_tokens)

    def test_vocab_guard_raises_on_tokenizer_mismatch(self) -> None:
        """Perplexity should fail fast if dataset token IDs exceed model vocab.

        This prevents meaningless 'comparisons' when the dataset was tokenized
        with a different tokenizer than the evaluated model.
        """
        # Create a dataset that includes a token ID outside the model vocab.
        bad_vocab_size = 16
        data = np.zeros(256, dtype=np.uint16)
        data[0] = np.uint16(bad_vocab_size + 1)  # out of range
        np.save(self.dataset_path, data)

        config = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=8,
            batch_size=1,
            num_batches=1,
        )
        model = DummyLMModel(vocab_size=bad_vocab_size)
        model.eval()

        benchmark = PerplexityBenchmark(config, self.device)
        with self.assertRaises(ValueError):
            _ = benchmark.run(model, "test_model")

    def test_valid_vocab_size_masks_padded_logits_for_fairness(self) -> None:
        """If model vocab is padded, valid_vocab_size should slice logits for fair CE.

        We construct a model whose padded logits dominate probability mass. Without
        slicing, CE is huge; with slicing to valid vocab, CE matches ln(V_valid).
        """
        import numpy as np

        valid_vocab = 16
        padded_vocab = 32
        # Tokens are all within valid vocab.
        data = np.random.randint(0, valid_vocab, size=512, dtype=np.uint16)
        np.save(self.dataset_path, data)

        class PaddedLogitsModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = padded_vocab

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, t = x.shape
                logits = torch.zeros((b, t, padded_vocab), dtype=torch.float32, device=x.device)
                # Make padded region overwhelmingly likely if not masked.
                logits[..., valid_vocab:] = 100.0
                return logits

        model = PaddedLogitsModel().eval()

        cfg = PerplexityBenchmarkConfig(
            dataset=str(self.dataset_path),
            block_size=32,
            batch_size=2,
            num_batches=2,
            valid_vocab_size=valid_vocab,
        )
        r = PerplexityBenchmark(cfg, self.device).run(model, "m")

        # With logits sliced to valid vocab and uniform over valid tokens => loss ≈ ln(V_valid).
        self.assertAlmostEqual(r.loss, math.log(valid_vocab), places=3)


if __name__ == "__main__":
    unittest.main()
