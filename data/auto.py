"""Automatic dataset builder

Selects the appropriate dataset implementation based on file extension, making
it easy to load different data formats without manually choosing the right class.
This factory pattern keeps data loading code simple and extensible.
"""
from __future__ import annotations

from typing import cast

from torch.utils.data import Dataset

from data.config import DatasetConfig, DatasetType
from data.npy import NpyDataset
from console.logger import Logger
from runtime.tensordict_utils import TensorDictBase

logger: Logger = Logger()


class AutoDataset(Dataset):
    """Auto dataset builder

    Selects the appropriate dataset implementation based on file extension, making
    it easy to load different data formats without manually choosing the right class.
    This factory pattern keeps data loading code simple and extensible.
    """
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._dataset: Dataset[TensorDictBase] | None = None

    def build(self) -> Dataset[TensorDictBase]:
        """Build token dataset from path

        Inspects the file extension and returns the appropriate dataset class,
        eliminating the need to know which dataset type handles which format.
        """
        match self.config.type:
            case DatasetType.NPY:
                return NpyDataset(self.config.source, block_size=self.config.tokens)
            case _:
                raise ValueError(f"Unsupported dataset type {self.config.type!r}")

    def __len__(self) -> int:
        """Get dataset length."""
        if self._dataset is None:
            self._dataset = self.build()
        # Type cast: we know build() returns NpyDataset which implements __len__
        dataset = cast(NpyDataset, self._dataset)
        return len(dataset)

    def __getitem__(self, index: int) -> TensorDictBase:
        """Get item by index."""
        if self._dataset is None:
            self._dataset = self.build()
        return self._dataset[index]

