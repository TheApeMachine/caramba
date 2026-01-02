"""Synthetic datasets for MOSAIC curriculum training.

This module provides a cheap, infinite-ish source of sequences that *require*
explicit memory over long gaps.

It is designed to support Stage D1 (teacher-forced memory addressing/gating):
- emits next-token training pairs (input_ids, target_ids)
- emits teacher signals:
  - mosaic_teacher_write_gate
  - mosaic_teacher_write_bucket
  - mosaic_teacher_read_bucket

The sequences are purely token-id based (no text/tokenizer dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict

import torch
from torch.utils.data import Dataset

from caramba.runtime.tensordict_utils import TensorDictBase


class _MosaicMemoryCurriculumDataset(Dataset[TensorDictBase]):
    def __init__(
        self,
        *,
        n_items: int,
        block_size: int,
        vocab_size: int,
        mem_buckets: int,
        mem_hashes: int,
        n_pairs: int,
        distractor_len: int,
        seed: int,
    ) -> None:
        self.n_items = int(n_items)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.mem_buckets = int(mem_buckets)
        self.mem_hashes = int(mem_hashes)
        self.n_pairs = int(n_pairs)
        self.distractor_len = int(distractor_len)
        self.seed = int(seed)

        # Reserve a tiny synthetic "protocol" token set in the low ids.
        self.T_SET = 1
        self.T_IS = 2
        self.T_GET = 3
        self.T_Q = 4
        self.T_PAD = 0

        self.min_tok = 8
        if self.vocab_size <= self.min_tok + 8:
            raise ValueError("vocab_size too small for synthetic curriculum")

    def __len__(self) -> int:
        return self.n_items

    def _bucket_for_key(self, key_tok: int) -> int:
        # Deterministic teacher addressing: stable mapping key->bucket.
        return int(key_tok) % int(self.mem_buckets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(idx))

        # Create key-value bindings.
        keys: list[int] = []
        vals: list[int] = []
        for _ in range(max(1, int(self.n_pairs))):
            k = rng.randrange(self.min_tok, self.vocab_size)
            v = rng.randrange(self.min_tok, self.vocab_size)
            keys.append(k)
            vals.append(v)

        seq: list[int] = []
        write_gate: list[int] = []
        write_utility: list[int] = []
        read_bucket: list[list[int]] = []
        write_bucket: list[list[int]] = []
        drop_local: list[int] = []

        def _append(
            tok: int,
            *,
            # By default, supervise "do not write" and "not useful" everywhere.
            # This prevents the write gate from drifting to high write rates due to
            # only seeing positive supervision on a tiny subset of tokens.
            wg: int = 0,
            wu: int = 0,
            rb: list[int] | None = None,
            wb: list[int] | None = None,
            dl: int = 0,
        ) -> None:
            seq.append(int(tok))
            write_gate.append(int(wg))
            write_utility.append(int(wu))
            if rb is None:
                read_bucket.append([-1] * int(self.mem_hashes))
            else:
                read_bucket.append([int(x) for x in rb])
            if wb is None:
                write_bucket.append([-1] * int(self.mem_hashes))
            else:
                write_bucket.append([int(x) for x in wb])
            drop_local.append(int(dl))

        # Write phase: "SET k IS v" with long distractors.
        for k, v in zip(keys, vals):
            b = self._bucket_for_key(k)
            bvec = [b] * int(self.mem_hashes)
            _append(self.T_SET, wg=0, wu=0)
            _append(k, wg=0, wu=0)
            _append(self.T_IS, wg=0, wu=0)
            # Write on value token (teacher-forced).
            _append(v, wg=1, wu=1, wb=bvec)
            # Distractors.
            for _ in range(int(self.distractor_len)):
                _append(rng.randrange(self.min_tok, self.vocab_size), wg=0, wu=0)

        # Query phase: "GET k ? IS v"
        for k, v in zip(keys, vals):
            b = self._bucket_for_key(k)
            bvec = [b] * int(self.mem_hashes)
            # Force reliance on memory/state throughout the query prefix.
            _append(self.T_GET, wg=0, wu=0, dl=1)
            _append(k, wg=0, wu=0, dl=1)
            _append(self.T_Q, wg=0, wu=0, dl=1)
            # On the token before emitting v as the next token (here: T_IS),
            # we want a memory read to be available.
            _append(self.T_IS, wg=0, wu=0, rb=bvec, dl=1)
            _append(v, wg=0, wu=0)
            # More distractors (small).
            for _ in range(max(0, int(self.distractor_len) // 4)):
                _append(rng.randrange(self.min_tok, self.vocab_size), wg=0, wu=0)

        # Pad/truncate to block_size+1 so we can create input/target pairs.
        # We need at least 2 tokens.
        need = int(self.block_size) + 1
        if len(seq) < need:
            for _ in range(need - len(seq)):
                _append(self.T_PAD)
        else:
            seq = seq[:need]
            write_gate = write_gate[:need]
            write_utility = write_utility[:need]
            read_bucket = read_bucket[:need]
            write_bucket = write_bucket[:need]
            drop_local = drop_local[:need]

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        # Teacher signals align to input token positions (same T as x/y).
        wg = torch.tensor(write_gate[:-1], dtype=torch.float32)
        wu = torch.tensor(write_utility[:-1], dtype=torch.float32)
        rb = torch.tensor(read_bucket[:-1], dtype=torch.long)
        wb = torch.tensor(write_bucket[:-1], dtype=torch.long)
        dl = torch.tensor(drop_local[:-1], dtype=torch.float32)

        return {
            "input_ids": x,
            "target_ids": y,
            "mosaic_teacher_write_gate": wg,
            "mosaic_teacher_write_utility": wu,
            "mosaic_teacher_read_bucket": rb,
            "mosaic_teacher_write_bucket": wb,
            "mosaic_drop_local": dl,
        }


@dataclass(frozen=True, slots=True)
class MosaicMemoryCurriculumDataset:
    """Manifest component wrapper for the synthetic curriculum dataset."""

    block_size: int = 512
    vocab_size: int = 8192
    mem_buckets: int = 16384
    mem_hashes: int = 2
    n_pairs: int = 1
    distractor_len: int = 64
    n_items: int = 100_000
    seed: int = 1337

    def build(self) -> Dataset[TensorDictBase]:
        return _MosaicMemoryCurriculumDataset(
            n_items=int(self.n_items),
            block_size=int(self.block_size),
            vocab_size=int(self.vocab_size),
            mem_buckets=int(self.mem_buckets),
            mem_hashes=int(self.mem_hashes),
            n_pairs=int(self.n_pairs),
            distractor_len=int(self.distractor_len),
            seed=int(self.seed),
        )

