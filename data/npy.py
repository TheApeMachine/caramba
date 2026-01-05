"""NPY dataset for preprocessed token data.

Training on raw text is slow because tokenization happens on-the-fly. For
efficiency, we preprocess text into token IDs and save them as .npy files.
This dataset loads those files and serves fixed-length token blocks for
next-token prediction training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict

_INT32_MAX = 2**31 - 1


class NpyDataset(Dataset[TensorDictBase]):
    """Dataset that loads preprocessed tokens from a .npy file.

    The file should contain a 1D array of token IDs. The dataset serves
    fixed-length blocks where each sample is (x, y) with y being the
    next-token shift of x—the standard format for language modeling.
    """

    def __init__(self, path: str, *, block_size: int) -> None:
        """Load tokens from a .npy file.

        Args:
            path: Path to the .npy file containing token IDs
            block_size: Length of token sequences to serve (context length)

        The file is memory-mapped for efficiency with large datasets.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        # Use mmap_mode="r" to avoid materializing large arrays into RAM.
        # We type hint this explicitly as an ndarray (which memmap is).
        arr = np.load(str(path), mmap_mode="r")
        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected numpy.load to return a numpy ndarray/memmap")

        # Reshape to 1D
        arr = arr.reshape(-1)

        # Suppress PyTorch's warning about non-writable tensors. The warning is
        # about potential undefined behavior if the tensor is written to, but
        # this dataset is read-only (we only slice in __getitem__). Copying
        # would defeat memory-mapping for datasets that don't fit in RAM.
        #
        # Update: We store the numpy array directly to avoid global PyTorch conversion issues
        # (e.g. uint16 support or forced copies). We convert small blocks on the fly.
        self.tokens_np: np.ndarray = arr
        self.block_size = int(block_size)

        # Validate a sample to ensure we aren't pointing at garbage.
        # We use numpy for sampling here to avoid materializing the whole array.
        self._validate_sample()

    def _validate_sample(self) -> None:
        """Sample tokens using numpy to verify range."""
        if self.tokens_np.size == 0:
            raise ValueError("Token array is empty.")

        # Check first/last/random 10k.
        #
        # For small arrays, sample *all* elements deterministically to avoid
        # flakiness in tests (random sampling can miss a rare invalid token).
        n = 10_000
        if int(self.tokens_np.size) <= n:
            indices = np.arange(int(self.tokens_np.size))
        else:
            indices = np.random.randint(0, int(self.tokens_np.size), size=int(n))
        sample = self.tokens_np[indices]

        mn = int(sample.min())
        mx = int(sample.max())

        if mn < 0:
            raise ValueError(f"Token IDs must be non-negative, found min={mn}")
        if mx > 2**31 - 1:
            raise ValueError(f"Token IDs must fit in int32, found max={mx}")

    def __len__(self) -> int:
        """Return the number of full non-overlapping blocks."""
        # Stride = block_size for standard LM training (no overlap).
        return (self.tokens_np.size - 1) // self.block_size

    @override
    def __getitem__(self, idx: int) -> TensorDictBase:
        """Get a (input, target) pair for language modeling.

        Returns x[0:block_size] and y[1:block_size+1] where y is the
        next-token target for each position in x.
        """
        # Calculate start position for non-overlapping block
        start = idx * self.block_size
        end = start + self.block_size + 1

        # Slice from mmap (numpy handles uint16 correctly)
        block_np = self.tokens_np[start:end]

        # Convert to tensor (copy) and cast to long
        # numpy uint16 -> torch int32/int64 works fine for small arrays.
        block = torch.from_numpy(block_np).to(dtype=torch.long)

        x = block[:-1]
        y = block[1:]
        return as_tensordict({"input_ids": x, "target_ids": y})
