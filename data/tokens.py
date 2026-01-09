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
from torch.utils.data import Dataset
from datasets import IterableDatasetDict

from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.data.base import Dataset
from caramba.console.logger import Logger
from caramba.data.error import DataError, DataErrorType
from caramba.data.config import DatasetConfig
from datasets import load_dataset

logger: Logger = Logger()


class TokenDataset(Dataset):
    """Tokenized dataset

    Wraps tokenized data stored as numpy arrays for efficient next-token prediction
    training. Handles both local files and automatic download/preparation from
    HuggingFace datasets, making it easy to switch between cached and fresh data.
    """
    def __init__(self, config: DatasetConfig):
        """Initialize token dataset

        Stores the configuration that specifies where to find or how to prepare
        the tokenized data, keeping the dataset flexible across different sources
        and preparation strategies.
        """
        self.config = config

    def build(self) -> None:
        """Build the dataset

        Attempts to load from disk first for speed, then falls back to downloading
        if the file is missing. This two-step approach avoids unnecessary network
        calls when data is already available locally.
        """
        self.err = self.load()

        if self.err is not None and self.err.isError(
            DataErrorType.DATASET_LOAD_FAILED
        ):
            self.err = self.download()

    def stream(self) -> IterableDatasetDict | DataError:
        """Stream dataset from source

        Returns an iterable dataset that can be consumed without loading everything
        into memory at once. This is essential for large datasets that don't fit
        in RAM, allowing training to start while data is still being downloaded.
        """
        return self.builder


    def download(self) -> DataError | None:
        """Download dataset from HuggingFace

        Fetches the dataset using streaming mode so it can be processed incrementally
        without waiting for the entire download to complete. Returns an error if the
        download fails, allowing the caller to handle missing or inaccessible datasets.
        """
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
        """Load dataset metadata from disk

        Reads the dataset metadata file that tracks where the actual tokenized data
        is stored. This metadata approach separates configuration from data storage,
        making it easier to manage multiple datasets and their locations.
        """
        with open(Path(f"artifacts/datasets/{self.config.source}.json"), "r") as f:
            data = json.load(f)

        if data is None:
            return DataError(DataErrorType.DATASET_LOAD_FAILED)

        return None