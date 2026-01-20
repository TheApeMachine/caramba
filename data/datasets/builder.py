"""Token dataset builder

Builds token datasets from path and block size. This keeps construction logic
out of training code so higher-level modules can simply orchestrate.
"""
from __future__ import annotations

from pathlib import Path

from data.npy import NpyDataset
from runtime.tensordict_utils import TensorDictBase
from torch.utils.data import Dataset


class TokenDatasetBuilder:
    """Build token datasets from path and block size."""

    @staticmethod
    def build(*, path: Path | str, block_size: int) -> Dataset[TensorDictBase]:
        """Build a token dataset from a NumPy file path.

        Args:
            path: Path to the `.npy` file containing tokenized data
            block_size: Sequence length for each training sample

        Returns:
            A dataset instance that serves (input, target) token pairs
        """
        return NpyDataset(str(path), block_size=int(block_size))
