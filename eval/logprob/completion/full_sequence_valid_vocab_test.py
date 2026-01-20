from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from eval.logprob.completion.full_sequence import LogprobCompletionFullSequence
from eval.logprob.completion.windowed import LogprobCompletionWindowed


class _PaddedLogitsModel(nn.Module):
    """Returns logits where padded vocab dominates unless sliced."""

    def __init__(self, *, vocab_size: int, valid_vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.valid_vocab_size = int(valid_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        logits = torch.zeros((b, t, self.vocab_size), dtype=torch.float32, device=x.device)
        # Make padded region overwhelmingly likely if not sliced.
        logits[..., self.valid_vocab_size :] = 100.0
        return logits


class _AssertNoGradModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch.is_grad_enabled():
            raise RuntimeError("grad should be disabled during logprob scoring")
        b, t = x.shape
        return torch.zeros((b, t, self.vocab_size), dtype=torch.float32, device=x.device)


class TestLogprobCompletionValidVocab(unittest.TestCase):
    def test_full_sequence_slices_logits_to_valid_vocab(self) -> None:
        valid_vocab = 8
        padded_vocab = 16
        model = _PaddedLogitsModel(vocab_size=padded_vocab, valid_vocab_size=valid_vocab)

        scorer = LogprobCompletionFullSequence(
            model=model, device=torch.device("cpu"), valid_vocab_size=valid_vocab
        )
        # Any in-vocab completion token should have logp = -ln(valid_vocab) under uniform logits.
        lp = scorer.score(prompt_ids=[1, 2], completion_ids=[3, 4])
        self.assertAlmostEqual(lp, -2.0 * math.log(valid_vocab), places=4)

    def test_logprob_scoring_runs_with_grad_disabled(self) -> None:
        """Regression: avoid autograd/inference-mode tensor conflicts on Metal."""
        model = _AssertNoGradModel(vocab_size=16).eval()
        device = torch.device("cpu")

        full = LogprobCompletionFullSequence(model=model, device=device, valid_vocab_size=16)
        _ = full.score(prompt_ids=[1, 2], completion_ids=[3])

        win = LogprobCompletionWindowed(model=model, device=device, context_window=16, valid_vocab_size=16)
        _ = win.score_batch(prompt_ids=[1, 2], completions_ids=[[3], [4]])


if __name__ == "__main__":
    unittest.main()

