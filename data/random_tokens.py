"""Synthetic random-token dataset (no files required).

This dataset exists to make it easy to run *real* end-to-end training loops
without requiring external corpora. It's primarily intended for:

- UI demos (streaming train.jsonl to a frontend)
- smoke tests of training/instrumentation plumbing
- quick perf checks on CPU/GPU

The samples are deterministic per-index given a seed, so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase


class _RandomTokensTorchDataset(Dataset[TensorDictBase]):
    def __init__(self, *, vocab_size: int, block_size: int, length: int, seed: int) -> None:
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.length = int(length)
        self.seed = int(seed)

        if self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")

    def __len__(self) -> int:
        return int(self.length)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Deterministic sample based on seed + idx.
        g = torch.Generator()
        g.manual_seed(int(self.seed) + int(idx))

        # Create a sequence of length block_size+1, then shift for next-token labels.
        seq = torch.randint(
            0,
            int(self.vocab_size),
            (int(self.block_size) + 1,),
            dtype=torch.long,
            generator=g,
        )
        x = seq[:-1].contiguous()
        y = seq[1:].contiguous()
        return {"input_ids": x, "target_ids": y}


@dataclass(frozen=True, slots=True)
class RandomTokenDataset:
    """Manifest dataset component that yields random token sequences.

    Config:
    - vocab_size: size of token vocabulary
    - block_size: sequence length (T)
    - length: number of samples in the dataset
    - seed: RNG seed used for deterministic per-index samples
    """

    vocab_size: int
    block_size: int
    length: int = 4096
    seed: int = 0

    def build(self) -> Dataset[TensorDictBase]:
        return _RandomTokensTorchDataset(
            vocab_size=int(self.vocab_size),
            block_size=int(self.block_size),
            length=int(self.length),
            seed=int(self.seed),
        )

