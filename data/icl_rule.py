"""In-context learning rule induction dataset

Generates few-shot learning tasks where models must infer rules from
demonstrations, with configurable gaps between examples and queries. Includes
binning labels for evaluating performance at different gap lengths, measuring
how well models generalize learned patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class _IclBuilder:
    """ICL sequence builder

    Constructs token sequences with bin labels, tracking which positions
    correspond to answer tokens that should be evaluated for accuracy at
    specific gap lengths.
    """
    tokens: list[int] = field(default_factory=list)
    bins: list[int] = field(default_factory=list)
    # Optional MOSAIC teacher signals (aligned to tokens list).
    write_gate: list[float] = field(default_factory=list)   # 0/1
    write_bucket: list[int] = field(default_factory=list)   # bucket index or -1
    read_bucket: list[int] = field(default_factory=list)    # bucket index or -1

    def append(
        self,
        tok: int,
        *,
        b: int = -1,
        wg: float = 0.0,
        wb: int = -1,
        rb: int = -1,
    ) -> None:
        """Append token with bin label

        Adds a token to the sequence along with its evaluation bin, where
        -1 means the position isn't evaluated and >=0 indicates which gap
        length bin this answer belongs to.
        """
        self.tokens.append(int(tok))
        self.bins.append(int(b))
        self.write_gate.append(float(wg))
        self.write_bucket.append(int(wb))
        self.read_bucket.append(int(rb))


class _IclRuleInductionTorchDataset(Dataset[dict[str, Tensor]]):
    """ICL rule induction dataset implementation

    Generates few-shot learning sequences with demonstrations, distractors, and
    queries, teaching models to infer rules from examples. Each sample includes
    bin labels for evaluating performance at different gap lengths.
    """
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
        emit_mem_teacher: bool,
        mem_buckets: int,
        query_from_demos: bool,
    ) -> None:
        """Initialize ICL rule induction dataset

        Sets up parameters for generating rule-learning tasks, including the
        number of demonstrations, gap lengths to evaluate, and distractors
        between examples to test robust pattern recognition.
        """
        self.n_items = int(n_items)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.n_demos = int(n_demos)
        self.gap_bins = [int(x) for x in gap_bins]
        self.demo_distractors = int(demo_distractors)
        self.seed = int(seed)
        self.emit_mem_teacher = bool(emit_mem_teacher)
        self.mem_buckets = int(mem_buckets)
        self.query_from_demos = bool(query_from_demos)

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
        if self.mem_buckets <= 0:
            raise ValueError(f"mem_buckets must be > 0, got {self.mem_buckets}")

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
        """Apply rule transformation

        Transforms an input token by adding a delta offset modulo the token
        range, creating a simple but learnable rule that models must infer
        from demonstrations.
        """
        # Keep values in [min_tok, vocab_size).
        span = int(self.vocab_size) - int(self.min_tok)
        if span <= 0:
            raise ValueError("Invalid token span")
        return int(self.min_tok) + int((int(x) - int(self.min_tok) + int(delta)) % span)

    def _bucket_for(self, x: int) -> int:
        """Map token -> teacher bucket index in [0, mem_buckets)."""
        return int((int(x) - int(self.min_tok)) % int(self.mem_buckets))

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

        # Sample demo inputs deterministically without replacement.
        available_pool_size = int(self.vocab_size) - int(self.min_tok)
        if int(self.n_demos) + 1 > available_pool_size:
            raise ValueError(
                f"Cannot sample {self.n_demos} demos + 1 query from pool of size {available_pool_size}. "
                f"Increase vocab_size or decrease n_demos."
            )

        # Draw n_demos unique tokens from the available pool
        pool = list(range(int(self.min_tok), int(self.vocab_size)))
        demos_x = rng.sample(pool, int(self.n_demos))
        used = set(demos_x)

        # Query token:
        # - If query_from_demos=True (default): query asks about one of the demonstrated keys.
        #   This is the classic "ICL-style associative recall" setup that MOSAIC memory is meant to solve.
        # - If query_from_demos=False: query asks about an unseen key (harder "rule induction" generalization).
        if bool(self.query_from_demos):
            xq = rng.choice(demos_x)
        else:
            remaining_pool = [x for x in pool if x not in used]
            xq = rng.sample(remaining_pool, 1)[0]
        yq = self._apply_rule(xq, delta=int(delta))

        bld = _IclBuilder()

        # Demonstrations: DEMO x IS y.
        for x in demos_x:
            y = self._apply_rule(x, delta=int(delta))
            bld.append(self.T_DEMO)
            bld.append(x)
            # Write at the "IS" position: it predicts y next.
            if self.emit_mem_teacher:
                bld.append(self.T_IS, wg=1.0, wb=self._bucket_for(int(x)))
            else:
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
        # Read at the "?" position: it predicts yq next.
        if self.emit_mem_teacher:
            bld.append(self.T_Q, b=int(bin_idx), rb=self._bucket_for(int(xq)))
        else:
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
            bld.write_gate = bld.write_gate[:need]
            bld.write_bucket = bld.write_bucket[:need]
            bld.read_bucket = bld.read_bucket[:need]

        x = torch.tensor(bld.tokens[:-1], dtype=torch.long)
        y = torch.tensor(bld.tokens[1:], dtype=torch.long)
        tb = torch.tensor(bld.bins[:-1], dtype=torch.long)

        # Loss masking:
        # This curriculum is intended to supervise *rule application* tokens, not random distractors
        # or trailing padding. We therefore mask targets everywhere except:
        # - demonstration "IS" positions (write_gate=1): model must predict y next
        # - query "?" positions (table2_bin>=0): model must predict yq next (Table 2 accuracy)
        #
        # Trainers/objectives use ignore_index=-100 by default.
        try:
            wg = torch.tensor(bld.write_gate[:-1], dtype=torch.float32)
            supervise = (wg > 0.5) | (tb >= 0)
            if supervise.shape == y.shape:
                y = y.clone()
                y[~supervise] = -100
        except Exception:
            # Never fail data generation on masking; worst-case we fall back to dense targets.
            pass
        out: dict[str, Tensor] = {
            "input_ids": x,
            "target_ids": y,
            "table2_bin": tb,
        }
        if self.emit_mem_teacher:
            out["memblock_teacher_write_gate"] = torch.tensor(bld.write_gate[:-1], dtype=torch.float32)
            out["memblock_teacher_write_bucket"] = torch.tensor(bld.write_bucket[:-1], dtype=torch.long)
            out["memblock_teacher_read_bucket"] = torch.tensor(bld.read_bucket[:-1], dtype=torch.long)
        return out


@dataclass(frozen=True, slots=True)
class IclRuleInductionDataset:
    """ICL rule induction dataset component

    Manifest-level dataset for few-shot rule learning experiments. Generates
    demonstration-query pairs with configurable gaps, enabling evaluation of
    how well models generalize learned patterns across different distances.
    """

    block_size: int = 512
    vocab_size: int = 8192
    n_items: int = 100_000
    seed: int = 1337
    n_demos: int = 4
    gap_bins: list[int] = field(default_factory=lambda: [0, 16, 64, 256])
    demo_distractors: int = 8
    # Optional: emit MOSAIC teacher signals so memory telemetry + teacher forcing can work.
    emit_mem_teacher: bool = False
    mem_buckets: int = 4096
    # If True, query asks about a key that appeared in demonstrations (associative recall).
    # If False, query key is unseen (harder rule induction generalization).
    query_from_demos: bool = True

    def build(self) -> Dataset[dict[str, Tensor]]:
        """Build ICL rule induction dataset

        Creates the PyTorch dataset that generates few-shot learning sequences
        with rule demonstrations and queries, ready for training and evaluation.
        """
        return _IclRuleInductionTorchDataset(
            n_items=int(self.n_items),
            block_size=int(self.block_size),
            vocab_size=int(self.vocab_size),
            n_demos=int(self.n_demos),
            gap_bins=list(self.gap_bins),
            demo_distractors=int(self.demo_distractors),
            seed=int(self.seed),
            emit_mem_teacher=bool(self.emit_mem_teacher),
            mem_buckets=int(self.mem_buckets),
            query_from_demos=bool(self.query_from_demos),
        )

