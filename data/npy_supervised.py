"""Simple supervised dataset from NumPy arrays.

This is intentionally small and dependency-light so non-language-model research
(classic ML, MLP baselines, etc.) can still be manifest-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase, as_tensordict

class _NpyPairDataset(Dataset[TensorDictBase]):
    def __init__(self, *, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) != len(y):
            raise ValueError(f"x and y length mismatch: {len(x)} != {len(y)}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> TensorDictBase:
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return as_tensordict({"inputs": x, "targets": y})


@dataclass(frozen=True, slots=True)
class NpySupervisedDataset:
    """Dataset component loading `x.npy` and `y.npy`.

    Config:
    - x_path: path to features array
    - y_path: path to targets/labels array
    """

    x_path: str
    y_path: str
    mmap: bool = True

    def build(self) -> Dataset[TensorDictBase]:
        x = np.load(Path(self.x_path), mmap_mode="r" if self.mmap else None)
        y = np.load(Path(self.y_path), mmap_mode="r" if self.mmap else None)
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Expected np.ndarray for x and y")
        return _NpyPairDataset(x=x, y=y)

