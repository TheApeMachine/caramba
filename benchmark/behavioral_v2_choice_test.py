from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from benchmark.behavioral_v2 import BenchmarkBehavioralV2
from config.benchmark import BehavioralV2BenchmarkConfig


class _FakeTokenizer:
    def __init__(self) -> None:
        self.n_vocab = 8
        self._map = {" A": 1, " B": 2, " C": 3}

    def encode(self, text: str) -> list[int]:
        s = str(text)
        if s in self._map:
            return [self._map[s]]
        # prompt => fixed token
        return [0]

    def decode(self, ids) -> str:
        return str(ids)


class _OneStepModel(nn.Module):
    """Produces logits that prefer token id 2."""

    def __init__(self, vocab: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        logits = torch.zeros((b, t, self.vocab_size), dtype=torch.float32, device=x.device)
        # Make token 2 best at all positions
        logits[..., 2] = 2.0
        logits[..., 1] = 1.0
        logits[..., 3] = 0.0
        return logits


class TestBehavioralV2ChoiceLogprob(unittest.TestCase):
    def test_choice_pick_and_margin(self) -> None:
        bb = BenchmarkBehavioralV2.__new__(BenchmarkBehavioralV2)
        bb.config = BehavioralV2BenchmarkConfig(seed=1, tests_per_category=1, max_new_tokens=2)
        bb.device = torch.device("cpu")
        bb.tokenizer = _FakeTokenizer()

        model = _OneStepModel().eval()
        picked, margin = bb._choice_pick_and_margin(  # type: ignore[attr-defined]
            model=model,
            prompt_ids=[0],
            choices=[" A", " B", " C"],
            expected=" B",
        )
        self.assertEqual(picked, " B")
        self.assertIsNotNone(margin)
        # margin should be > 0 since expected is also the argmax.
        assert margin is not None
        self.assertGreater(margin, 0.0)
        # Roughly, correct(2.0) - best_other(1.0) in logprob space (not raw logit),
        # should still be positive.


if __name__ == "__main__":
    unittest.main()

