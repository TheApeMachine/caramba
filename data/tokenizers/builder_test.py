from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from caramba.data.config import DatasetConfig, DatasetType
from caramba.data.tokenizers.builder import TokenizerBuilder
from caramba.data.error import DataErrorType

def test_builder_build_requires_config():
    builder = TokenizerBuilder(config=None)
    with pytest.raises(ValueError, match="Cannot build dataset without config"):
        builder.build()

def test_builder_build_requires_block_size():
    config = DatasetConfig(
        type=DatasetType.TOKENS,
        source="test",
        tokens=100,
        tokenizer="tiktoken:gpt2"
        # block_size missing (defaults to None)
    )
    builder = TokenizerBuilder(config=config)
    with pytest.raises(ValueError, match="block_size must be specified"):
        builder.build()

def test_builder_build_local_npy(tmp_path):
    # Test handling of .npy source
    npy_file = tmp_path / "data.npy"
    npy_file.touch()
    
    config = DatasetConfig(
        type=DatasetType.NPY,
        source=str(npy_file),
        tokens=100,
        block_size=128
    )
    builder = TokenizerBuilder(config=config)
    
    with patch("caramba.data.npy.NpyDataset") as MockNpy:
        dataset = builder.build()
        MockNpy.assert_called_once_with(str(npy_file), block_size=128)

def test_builder_build_cached_artifact(tmp_path):
    # Test handling of existing cached artifact
    config = DatasetConfig(
        type=DatasetType.TOKENS,
        source="cached_data",
        tokens=100,
        block_size=128
    )
    builder = TokenizerBuilder(config=config)
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("caramba.data.npy.NpyDataset") as MockNpy:
            dataset = builder.build()
            # Should look in artifacts/datasets/cached_data.npy
            MockNpy.assert_called_once()
            args = MockNpy.call_args
            # Path is artifacts/datasets/cached_data/cached_data_100.npy
            assert "artifacts/datasets/cached_data/cached_data_100.npy" in args[0][0]

@patch("caramba.data.tokenizers.builder.load_dataset")
def test_builder_tokenize_and_save(mock_load_dataset):
    # Test full flow of tokenize and save
    config = DatasetConfig(
        type=DatasetType.TOKENS,
        source="remote_data",
        tokens=10,
        block_size=128,
        tokenizer="tiktoken:gpt2"
    )
    builder = TokenizerBuilder(config=config)
    
    # Mock dataset
    mock_split = [{"text": "hello world"}]
    mock_load_dataset.return_value = {
        "train": mock_split,
        "validation": [],
        "test": []
    }
    
    with patch("numpy.save") as mock_save:
         with patch("pathlib.Path.mkdir"):
            # We also need to avoid NpyDataset trying to load the file, 
            # so we mock exists to return False first (triggering build).
            # Then parent.exists() is called inside _tokenize_and_save (return True).
            # Then exists() is called again after build (return True to trigger load).
            with patch("pathlib.Path.exists", side_effect=[False, True, True]):
                with patch("caramba.data.npy.NpyDataset") as MockNpy:
                    dataset = builder.build()
                    MockNpy.assert_called_once()
                    mock_save.assert_called_once()

def test_builder_fails_without_tokenizer_in_config():
    config = DatasetConfig(
        type=DatasetType.TOKENS,
        source="remote_data",
        tokens=10,
        block_size=128,
        tokenizer=None # Missing
    )
    builder = TokenizerBuilder(config=config)
    
    with patch("pathlib.Path.exists", return_value=False):
        # build() -> _tokenize_and_save() -> error
        # It logs error and returns DataError, then build raises ValueError because it failed to build.
        # Or _tokenize_and_save returns DataError, so npy_path.exists is checked again (false),
        # so it raises ValueError "Could not build dataset..."
        with pytest.raises(ValueError, match="Could not build dataset"):
            builder.build()
