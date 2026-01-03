"""ICL-like synthetic rule induction dataset.

Row D (Table 2): Few-shot rule induction with distractors, sweeping the gap
between demonstrations and query.

The dataset emits:
- input_ids, target_ids: next-token LM pairs
- table2_bin: (T,) int64 bin id for evaluation positions, -1 elsewhere

`table2_bin` is aligned to input positions: at positions where table2_bin>=0,
the next token (target_ids at the same position) is the answer token whose
accuracy should be attributed to that bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class _IclBuilder:
    tokens: list[int] = field(default_factory=list)
    bins: list[int] = field(default_factory=list)

    def append(self, tok: int, *, b: int = -1) -> None:
        self.tokens.append(int(tok))
        self.bins.append(int(b))


class _IclRuleInductionTorchDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        n_items: int,
        block_size: int,
        vocab_size: int,
        n_demos: int,
        gap_bins: list[int],
        demo_distractors: int,
        seed: int,
    ) -> None:
        self.n_items = int(n_items)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.n_demos = int(n_demos)
        self.gap_bins = [int(x) for x in gap_bins]
        self.demo_distractors = int(demo_distractors)
        self.seed = int(seed)

        if self.n_items <= 0:
            raise ValueError(f"n_items must be > 0, got {self.n_items}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.vocab_size <= 16:
            raise ValueError(f"vocab_size too small, got {self.vocab_size}")
        if self.n_demos < 1:
            raise ValueError(f"n_demos must be >= 1, got {self.n_demos}")
        if not self.gap_bins:
            raise ValueError("gap_bins must be non-empty")
        if any(x < 0 for x in self.gap_bins):
            raise ValueError("gap_bins must be >= 0")
        if self.demo_distractors < 0:
            raise ValueError(f"demo_distractors must be >= 0, got {self.demo_distractors}")

        # Reserve low tokens as protocol markers.
        self.T_PAD = 0
        self.T_DEMO = 1
        self.T_QUERY = 2
        self.T_IS = 3
        self.T_Q = 4

        self.min_tok = 8
        if self.vocab_size <= self.min_tok + 16:
            raise ValueError("vocab_size too small for rule-induction curriculum")

    def __len__(self) -> int:
        return int(self.n_items)

    def _apply_rule(self, x: int, *, delta: int) -> int:
        # Keep values in [min_tok, vocab_size).
        span = int(self.vocab_size) - int(self.min_tok)
        if span <= 0:
            raise ValueError("Invalid token span")
        return int(self.min_tok) + int((int(x) - int(self.min_tok) + int(delta)) % span)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rng = random.Random(int(self.seed) + int(idx))

        # Per-sample rule parameter (additive offset mod span).
        span = int(self.vocab_size) - int(self.min_tok)
        if span < 2:
            raise ValueError("vocab_size too small for rule parameterization")
        delta = rng.randrange(1, min(span, 33))

        # Choose gap bin for this sample.
        bin_idx = rng.randrange(0, len(self.gap_bins))
        gap_len = int(self.gap_bins[bin_idx])

        # Sample demo inputs.
        demos_x: list[int] = []
        used: set[int] = set()
        for _ in range(int(self.n_demos)):
            x = rng.randrange(int(self.min_tok), int(self.vocab_size))
            while x in used:
                x = rng.randrange(int(self.min_tok), int(self.vocab_size))
            used.add(x)
            demos_x.append(x)

        # Query input should be unseen in demos.
        xq = rng.randrange(int(self.min_tok), int(self.vocab_size))
        while xq in used:
            xq = rng.randrange(int(self.min_tok), int(self.vocab_size))
        yq = self._apply_rule(xq, delta=int(delta))

        bld = _IclBuilder()

        # Demonstrations: DEMO x IS y.
        for x in demos_x:
            y = self._apply_rule(x, delta=int(delta))
            bld.append(self.T_DEMO)
            bld.append(x)
            bld.append(self.T_IS)
            bld.append(y)
            for _ in range(int(self.demo_distractors)):
                bld.append(rng.randrange(int(self.min_tok), int(self.vocab_size)))

        # Gap distractors.
        for _ in range(int(gap_len)):
            bld.append(rng.randrange(int(self.min_tok), int(self.vocab_size)))

        # Query: QUERY x ? y
        bld.append(self.T_QUERY)
        bld.append(xq)
        bld.append(self.T_Q, b=int(bin_idx))  # attribute accuracy of next token to this gap bin
        bld.append(yq)

        # Pad/truncate to block_size+1.
        need = int(self.block_size) + 1
        if len(bld.tokens) < need:
            for _ in range(need - len(bld.tokens)):
                bld.append(self.T_PAD, b=-1)
        else:
            bld.tokens = bld.tokens[:need]
            bld.bins = bld.bins[:need]

        x = torch.tensor(bld.tokens[:-1], dtype=torch.long)
        y = torch.tensor(bld.tokens[1:], dtype=torch.long)
        tb = torch.tensor(bld.bins[:-1], dtype=torch.long)
        return {
            "input_ids": x,
            "target_ids": y,
            "table2_bin": tb,
        }


@dataclass(frozen=True, slots=True)
class IclRuleInductionDataset:
    """Manifest dataset component wrapper."""

    block_size: int = 512
    vocab_size: int = 8192
    n_items: int = 100_000
    seed: int = 1337
    n_demos: int = 4
    gap_bins: list[int] = field(default_factory=lambda: [0, 16, 64, 256])
    demo_distractors: int = 8

    def build(self) -> Dataset[dict[str, Tensor]]:
        return _IclRuleInductionTorchDataset(
            n_items=int(self.n_items),
            block_size=int(self.block_size),
            vocab_size=int(self.vocab_size),
            n_demos=int(self.n_demos),
            gap_bins=list(self.gap_bins),
            demo_distractors=int(self.demo_distractors),
            seed=int(self.seed),
        )

