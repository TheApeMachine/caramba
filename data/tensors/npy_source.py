"""NumPy tensor source

Wraps a NumPy array (possibly memory-mapped) to provide tensor access
with automatic dtype promotion for unsupported types like uint16.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
from torch import Tensor

from data.tensors.source import TensorSource


class NpySource:
    """NumPy tensor source

    Wraps a NumPy array (possibly memory-mapped) to provide tensor access
    with automatic dtype promotion for unsupported types like uint16.
    """
    def __init__(self, arr: np.ndarray) -> None:
        """Initialize NumPy source

        Stores the array reference, which may be memory-mapped for large
        datasets that don't fit in RAM.
        """
        self.arr = arr

    def __len__(self) -> int:
        """Get number of samples

        Returns the first dimension size, which represents the number of
        samples in the dataset.
        """
        return int(self.arr.shape[0])

    def get(self, idx: int) -> Tensor:
        """Get tensor at index

        Converts a NumPy array slice to a PyTorch tensor, promoting uint16 to
        int32 since PyTorch has limited support for uint16 operations.
        """
        x = self.arr[idx]
        # Suppress PyTorch warning about read-only arrays since we only read from them
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            t = torch.from_numpy(np.asarray(x))
        # PyTorch does not implement many ops (e.g. index_select) for uint16.
        # Promote to a supported integer dtype at the data boundary.
        if t.dtype == torch.uint16:
            return t.to(torch.int32)
        return t
