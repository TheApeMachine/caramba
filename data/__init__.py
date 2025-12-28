"""Dataset utilities for training data loading.

Training requires feeding the model sequences of tokens. This package
provides Dataset implementations that load preprocessed token data
and serve it in the (input, target) pairs needed for language modeling.
"""
from __future__ import annotations

from data.auto import build_token_dataset
from data.npy import NpyDataset
from data.token_dataset import TokenDataset
from data.text_tokens import TextTokensDataset

__all__ = ["NpyDataset", "TextTokensDataset", "TokenDataset", "build_token_dataset"]
