"""NumPy token dataset

Loads pre-tokenized data from `.npy` files and serves fixed-length sequences
for next-token prediction. Using preprocessed tokens avoids repeated
tokenization overhead during training, significantly speeding up data loading.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict
from caramba.data.base import Dataset as CarambaDataset

_INT32_MAX = 2**31 - 1


class NpyDataset(Dataset[TensorDictBase], CarambaDataset):
    """NumPy token dataset

    Loads a 1D array of token IDs from disk and serves fixed-length blocks
    for next-token prediction. Uses memory-mapping to handle datasets larger
    than RAM without loading everything into memory at once.
    """

    def __init__(self, path: str, *, block_size: int) -> None:
        """Initialize NumPy dataset

        Opens the token file using memory-mapping so large datasets don't need
        to fit in RAM. The block_size determines the sequence length for each
        training sample, which should match your model's context window.
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
        """Validate token range

        Checks that token IDs are in a valid range (non-negative and fit in
        int32) by sampling from the dataset. This catches corrupted files or
        incorrect tokenization early rather than during training.
        """
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
        """Get dataset length

        Calculates how many non-overlapping blocks fit in the token array,
        ensuring each sample is independent and the dataset size is
        predictable for training loop progress tracking.
        """
        # Stride = block_size for standard LM training (no overlap).
        return (self.tokens_np.size - 1) // self.block_size

    @override
    def __getitem__(self, index: int) -> TensorDictBase:
        """Get training sample

        Extracts a token block and creates (input, target) pairs where target
        is the input shifted by one position. This is the standard format for
        autoregressive language modeling where the model predicts the next token.
        """
        # Calculate start position for non-overlapping block
        start = index * self.block_size
        end = start + self.block_size + 1

        # Slice from mmap (numpy handles uint16 correctly)
        block_np = self.tokens_np[start:end]

        # Convert to tensor and cast to long
        # numpy uint16 -> torch int32/int64 works fine for small arrays.
        # Suppress PyTorch warning about read-only arrays since this dataset is read-only.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            block = torch.from_numpy(block_np).to(dtype=torch.long)

        x = block[:-1]
        y = block[1:]
        return as_tensordict({"input_ids": x, "target_ids": y})
