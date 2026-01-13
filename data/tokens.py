"""Token dataset module

Provides the manifest-level entrypoint for tokenized `.npy` datasets used in
next-token prediction training. Supports both loading existing tokenized files
and automatically downloading/preparing datasets from HuggingFace, making it
easy to switch between cached local data and fresh downloads without changing
your training code.
"""
from __future__ import annotations

from typing import Any
from datasets import IterableDatasetDict
from caramba.data.tokenizers.builder import TokenizerBuilder
from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.data.base import Dataset
from caramba.console.logger import Logger
from caramba.data.error import DataError
from caramba.data.config import DatasetConfig, DatasetType

logger: Logger = Logger()


class TokenDataset(Dataset):
    """Tokenized dataset

    Wraps tokenized data stored as numpy arrays for efficient next-token prediction
    training. Handles both local files and automatic download/preparation from
    HuggingFace datasets.
    """
    def __init__(
        self,
        config: DatasetConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize token dataset

        Supports initialization via explicit kwargs (registry style) or DatasetConfig object.
        """
        if config is None:
            # Handle legacy/registry config where 'dataset' maps to 'source'
            if "dataset" in kwargs:
                kwargs["source"] = kwargs.pop("dataset")

            # Ensure type is set if missing
            if "type" not in kwargs:
                kwargs["type"] = DatasetType.TOKENS

            config = DatasetConfig(**kwargs)

        self.config = config
        self.tokenizer = TokenizerBuilder(config)
        self._dataset = None

    def build(self) -> Any:
        """Build and return the dataset

        Returns the underlying dataset implementation (e.g. NpyDataset) ready for use.
        """
        if self._dataset is None:
            self._dataset = self.tokenizer.build()
        return self._dataset

    def __len__(self) -> int:
        if self._dataset is None:
            self.build()
        if self._dataset is not None:
            return len(self._dataset)

        raise ValueError("Dataset not built")

    def __getitem__(self, index: int) -> TensorDictBase:
        if self._dataset is None:
            self.build()
        if self._dataset is not None:
            return self._dataset[index] # type: ignore

        raise ValueError("Dataset not built")

    def stream(self) -> IterableDatasetDict | DataError:
        """Stream dataset from source"""
        return self.tokenizer.stream()
