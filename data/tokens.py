"""Token dataset module

Provides the manifest-level entrypoint for tokenized `.npy` datasets used in
next-token prediction training. Supports both loading existing tokenized files
and automatically downloading/preparing datasets from HuggingFace, making it
easy to switch between cached local data and fresh downloads without changing
your training code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from datasets import IterableDatasetDict, load_dataset
from typing import Any

from runtime.tensordict_utils import TensorDictBase
from data.base import Dataset
from console.logger import Logger
from data.error import DataError, DataErrorType
from data.config import DatasetConfig
from datasets import load_dataset

logger: Logger = Logger()


class TokenDataset(Dataset):
    """Tokenized dataset

    Wraps tokenized data stored as numpy arrays for efficient next-token prediction
    training. Handles both local files and automatic download/preparation from
    HuggingFace datasets.
    """
    def __init__(
        self,
        path: str | None = None,
        block_size: int | None = None,
        tokenizer: str | None = None,
        source: str | None = None,
        type: str | None = None,
        tokens: int | None = None,
        config: DatasetConfig | None = None,
    ):
        """Initialize token dataset

        Supports initialization via explicit kwargs (registry style) or DatasetConfig object.
        """
        self.path = path
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Backward compatibility / alternative config
        if config is not None:
            self.config = config
            if self.path is None:
                self.path = config.source
            if self.block_size is None:
                self.block_size = config.tokens

        # Also handle registry passing 'source' instead of 'path' if applicable,
        # though herorun.yml passes 'path'.
        if self.path is None and source is not None:
            self.path = source

        self._dataset: Dataset | None = None

    def build(self) -> Any:
        """Build and return the dataset

        Returns the underlying dataset implementation (e.g. NpyDataset) ready for use.
        """
        if self._dataset is not None:
            return self._dataset

        if self.path and str(self.path).endswith(".npy"):
            from data.npy import NpyDataset
            if self.block_size is None:
                raise ValueError("block_size must be specified for NpyDataset")
            self._dataset = NpyDataset(self.path, block_size=int(self.block_size))
            return self._dataset

        # Fallback to existing logic for HF datasets / config-based loading
        if hasattr(self, "config") and self.config:
             # Existing logic adapted to return self or a dataset
             self.err = self.load()
             if self.err is not None and self.err.isError(DataErrorType.DATASET_LOAD_FAILED):
                 self.err = self.download()

             # If successful, self.builder should be set (it was in the original code,
             # though original code didn't implement __len__/__getitem__ on TokenDataset).
             # The original code's `stream()` returned `self.builder`.
             # We need to support map-style Access for StandardTrainer if possible.
             # But HF streaming datasets are IterableDatasets.
             if self.builder:
                 return self.builder

        raise ValueError(
            f"Could not build TokenDataset. path={self.path}, block_size={self.block_size}. "
            "Ensure path points to a .npy file or config is valid."
        )

    def __len__(self) -> int:
        if self._dataset is None:
            self.build()
        if self._dataset is not None:
            return len(self._dataset) # type: ignore
        raise ValueError("Dataset not built")

    def __getitem__(self, index: int) -> TensorDictBase:
        if self._dataset is None:
            self.build()
        if self._dataset is not None:
            return self._dataset[index] # type: ignore
        raise ValueError("Dataset not built")

    def stream(self) -> IterableDatasetDict | DataError:
        """Stream dataset from source"""
        # Kept for compatibility if used elsewhere
        return self.builder

    def download(self) -> DataError | None:
        """Download dataset from HuggingFace"""
        # Kept for compatibility
        if not hasattr(self, "config"):
             return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

        self.builder = load_dataset(
            path=self.config.source,
            name=None,
            split=None,
            streaming=True
        )

        if self.builder is None:
            return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

        return None

    def load(self) -> DataError | None:
        """Load dataset metadata from disk"""
        # Kept for compatibility
        if not hasattr(self, "config"):
             return DataError(DataErrorType.DATASET_LOAD_FAILED)

        try:
            with open(Path(f"artifacts/datasets/{self.config.source}.json"), "r") as f:
                data = json.load(f)
            if data is None:
                return DataError(DataErrorType.DATASET_LOAD_FAILED)
        except FileNotFoundError:
             return DataError(DataErrorType.DATASET_LOAD_FAILED)

        return None