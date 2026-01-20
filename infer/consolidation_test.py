from __future__ import annotations

import unittest

import torch
from torch import nn

from infer.consolidation import ReplayConsolidator
from infer.context import InferContext
from infer.event_runtime import StreamModelRunner
from infer.replay import ReplayBuffer


class _ToyLM(nn.Module):
    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.emb = nn.Embedding(self.vocab, 16)
        self.lin = nn.Linear(16, self.vocab)

    def forward(self, input_ids: torch.Tensor, *, ctx=None):  # type: ignore[override]
        x = self.emb(input_ids)
        return self.lin(x)


class ConsolidationTest(unittest.TestCase):
    def test_replay_consolidator_runs(self) -> None:
        model = _ToyLM(vocab=32)
        ctx = InferContext(caches=[])
        runner = StreamModelRunner(model=model, ctx=ctx, collect_aux=False)

        replay = ReplayBuffer(max_sequences=8, rng_seed=0)
        replay.add([1, 2, 3, 4, 5, 6, 7, 8, 9])

        cons = ReplayConsolidator(runner=runner, replay=replay, block_size=4, batch_size=2, online=None)
        m = cons.consolidate_once()
        self.assertIsNotNone(m)
        assert m is not None
        self.assertIn("replay/loss", m)

