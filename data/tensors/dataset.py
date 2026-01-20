"""Tensor files dataset implementation

Combines multiple tensor sources into a single dataset, ensuring all
sources have matching lengths and applying transforms to each sample.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase, as_tensordict
from data.transforms import Compose
from data.tensors.source import TensorSource


class TensorFilesDataset(Dataset[TensorDictBase]):
    """Tensor files dataset implementation

    Combines multiple tensor sources into a single dataset, ensuring all
    sources have matching lengths and applying transforms to each sample.
    """
    def __init__(self, *, sources: dict[str, TensorSource], transforms: Compose) -> None:
        """Initialize tensor files dataset

        Validates that all tensor sources have the same length, ensuring
        samples align correctly across different data modalities or features.
        """
        self.sources = sources
        self.transforms = transforms
        lengths = [len(src) for src in sources.values()]
        if not lengths:
            raise ValueError("dataset.tensors requires at least one file mapping")
        if len(set(lengths)) != 1:
            raise ValueError(f"All tensor sources must share the same length, got {sorted(set(lengths))}")
        self._len = int(lengths[0])

    def __len__(self) -> int:
        """Get dataset length

        Returns the number of samples, which is the same across all tensor
        sources since they were validated to have matching lengths.
        """
        return int(self._len)

    def __getitem__(self, idx: int) -> TensorDictBase:
        """Get sample

        Fetches tensors from all sources at the given index and combines them
        into a TensorDict, then applies the transform pipeline to produce the
        final sample format expected by the model.
        """
        payload: dict[str, Any] = {k: src.get(idx) for k, src in self.sources.items()}
        td = as_tensordict(payload)
        return self.transforms(td)
