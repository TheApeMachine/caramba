"""Supervised NumPy dataset

Loads feature and target arrays from separate NumPy files for classic
supervised learning tasks. Minimal dependencies make it suitable for simple
ML experiments that don't need complex data pipelines.
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
    """NumPy pair dataset implementation

    Wraps two NumPy arrays (features and targets) into a PyTorch dataset,
    handling dtype conversions and ensuring arrays have matching lengths.
    """
    def __init__(self, *, x: np.ndarray, y: np.ndarray) -> None:
        """Initialize pair dataset

        Validates that feature and target arrays have the same number of
        samples, ensuring each feature vector has a corresponding target.
        """
        if len(x) != len(y):
            raise ValueError(f"x and y length mismatch: {len(x)} != {len(y)}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """Get dataset length

        Returns the number of samples, which should match for both arrays
        since they were validated during initialization.
        """
        return int(len(self.x))

    def __getitem__(self, idx: int) -> TensorDictBase:
        """Get sample pair

        Converts NumPy array slices to PyTorch tensors, promoting uint16 to
        int32 since PyTorch has limited support for uint16 operations.
        """
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        # Avoid uint16 tensors (many PyTorch ops don't support them).
        if x.dtype == torch.uint16:
            x = x.to(torch.long)
        if y.dtype == torch.uint16:
            y = y.to(torch.long)
        return as_tensordict({"inputs": x, "targets": y})


@dataclass(frozen=True, slots=True)
class NpySupervisedDataset:
    """Supervised NumPy dataset component

    Manifest-level dataset that loads features and targets from separate NumPy
    files, supporting memory-mapping for large arrays that don't fit in RAM.
    """

    x_path: str
    y_path: str
    mmap: bool = True

    def build(self) -> Dataset[TensorDictBase]:
        """Build supervised dataset

        Loads both NumPy files, optionally using memory-mapping, and creates
        a dataset that pairs corresponding feature and target samples.
        """
        x = np.load(Path(self.x_path), mmap_mode="r" if self.mmap else None)
        y = np.load(Path(self.y_path), mmap_mode="r" if self.mmap else None)
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Expected np.ndarray for x and y")
        return _NpyPairDataset(x=x, y=y)

