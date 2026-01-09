"""Random token dataset

Generates synthetic token sequences on-the-fly without requiring data files,
making it easy to test training infrastructure or run demos. Samples are
deterministic based on index and seed, ensuring reproducible experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase


class _RandomTokensTorchDataset(Dataset[TensorDictBase]):
    """Random tokens dataset implementation

    Generates random token sequences deterministically based on sample index,
    ensuring the same index always produces the same sequence for reproducible
    testing and debugging.
    """
    def __init__(self, *, vocab_size: int, block_size: int, length: int, seed: int) -> None:
        """Initialize random tokens dataset

        Sets up parameters for generating synthetic sequences, validating that
        all values are positive and reasonable for token generation.
        """
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
        """Get dataset length

        Returns the configured number of samples, which are generated on-demand
        rather than pre-computed to keep memory usage minimal.
        """
        return int(self.length)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Generate random sample

        Creates a deterministic random sequence by seeding the generator with
        the sample index, then shifts it to create (input, target) pairs for
        next-token prediction training.
        """
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
    """Random token dataset component

    Manifest-level dataset that generates synthetic token sequences without
    requiring data files. Useful for testing, demos, and quick performance
    benchmarks where real data isn't necessary.
    """

    vocab_size: int
    block_size: int
    length: int = 4096
    seed: int = 0

    def build(self) -> Dataset[TensorDictBase]:
        """Build random token dataset

        Creates the PyTorch dataset that will generate samples on-demand,
        making it memory-efficient even for large synthetic datasets.
        """
        return _RandomTokensTorchDataset(
            vocab_size=int(self.vocab_size),
            block_size=int(self.block_size),
            length=int(self.length),
            seed=int(self.seed),
        )

