"""Dataset components.

These are manifest-referenced datasets that can be built into torch Datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from data.auto import build_token_dataset


@dataclass(frozen=True, slots=True)
class TokenDataset:
    """Token dataset component for next-token training.

    Config:
    - path: dataset file path (.npy/.tokens/.txt)
    - block_size: sequence length
    """

    path: str
    block_size: int

    def build(self) -> Dataset[tuple[Tensor, Tensor]]:
        return build_token_dataset(path=Path(self.path), block_size=int(self.block_size))

