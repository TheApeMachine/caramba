from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, ANY

from caramba.data.tokens import TokenDataset
from caramba.data.config import DatasetConfig, DatasetType
from caramba.data.tokenizers.builder import TokenizerBuilder

def test_tokendataset_build_delegates_to_builder():
    # Verify TokenDataset delegates build() to TokenizerBuilder
    config = DatasetConfig(
        type=DatasetType.TOKENS, 
        source="testsource", 
        tokens=100,
        tokenizer="tiktoken:gpt2",
        block_size=128
    )
    dataset = TokenDataset(config=config)
    
    # Mock the builder instance inside the dataset
    dataset.tokenizer = MagicMock(spec=TokenizerBuilder)
    
    dataset.build()
    
    dataset.tokenizer.build.assert_called_once()

def test_tokendataset_stream_delegates_to_builder():
    # Verify TokenDataset delegates stream() to TokenizerBuilder
    config = DatasetConfig(
        type=DatasetType.TOKENS, 
        source="testsource", 
        tokens=100,
        tokenizer="tiktoken:gpt2",
        block_size=128
    )
    dataset = TokenDataset(config=config)
    
    # Mock the builder instance
    dataset.tokenizer = MagicMock(spec=TokenizerBuilder)
    
    dataset.stream()
    
    dataset.tokenizer.stream.assert_called_once()
