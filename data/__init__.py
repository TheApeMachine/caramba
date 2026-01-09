"""Data package

Provides dataset implementations that load preprocessed token data and serve
it in (input, target) pairs for language modeling. This package abstracts away
the complexity of data loading so training code can focus on model architecture
and optimization.
"""
from __future__ import annotations

from caramba.data.auto import AutoDataset
from caramba.data.npy import NpyDataset
from caramba.data.tokens import TokenDataset
from caramba.data.datasets.builder import TokenDatasetBuilder

__all__ = ["NpyDataset", "TokenDataset", "AutoDataset", "TokenDatasetBuilder"]
