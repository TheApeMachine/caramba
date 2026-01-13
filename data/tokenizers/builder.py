"""Tokenizer builder

Builds tokenizers from config objects. This keeps construction logic out of
benchmarks/eval code so higher-level modules can simply orchestrate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from datasets import load_dataset, IterableDatasetDict
from caramba.config.eval import (
    CodeBpeTokenizerConfig,
    LlamaTokenizerConfig,
    TiktokenTokenizerConfig,
    TokenizerConfig,
)
from caramba.data.tokenizers.base import Tokenizer
from caramba.data.tokenizers.bpe_eval import CodeBpeEvalTokenizer
from caramba.data.tokenizers.huggingface import HuggingfaceTokenizer
from caramba.data.tokenizers.tiktoken import TiktokenTokenizer
from caramba.data.config import DatasetConfig
from caramba.data.error import DataError, DataErrorType
from caramba.console.logger import Logger
from caramba.data.base import Dataset

logger: Logger = Logger()

class TokenizerBuilder:
    """Build tokenizers from `caramba.config.eval.TokenizerConfig`."""

    def __init__(self, config: DatasetConfig | None = None):
        self.config = config
        self.tokenizer = None

    def _get_tokenizer_instance(self, cfg: TokenizerConfig) -> Tokenizer:
        if isinstance(cfg, TiktokenTokenizerConfig):
            return TiktokenTokenizer(encoding=str(cfg.encoding))
        if isinstance(cfg, LlamaTokenizerConfig):
            return HuggingfaceTokenizer(model_id=str(cfg.model_id))
        if isinstance(cfg, CodeBpeTokenizerConfig):
            return CodeBpeEvalTokenizer(tokenizer_file=str(cfg.tokenizer_file))
        raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")

    def build_tokenizer(self, cfg: TokenizerConfig) -> Tokenizer:
        return self._get_tokenizer_instance(cfg)

    def build(self) -> Dataset:
        """Build and return the dataset implementation."""
        if self.config is None:
            raise ValueError("Cannot build dataset without config")
            
        block_size = self.config.block_size
        if block_size is None:
             raise ValueError("block_size must be specified in DatasetConfig")

        # 1. Check for local .npy file
        if self.config.source.endswith(".npy"):
             from caramba.data.npy import NpyDataset
             return NpyDataset(self.config.source, block_size=block_size)

        # 2. Check for pre-tokenized artifact
        npy_path = self._get_artifact_path()
        if npy_path.exists():
            from caramba.data.npy import NpyDataset
            return NpyDataset(str(npy_path), block_size=block_size)

        # 3. Tokenize and save
        err = self._tokenize_and_save(npy_path)
        if err is None and npy_path.exists():
            from caramba.data.npy import NpyDataset
            return NpyDataset(str(npy_path), block_size=block_size)

        raise ValueError(f"Could not build dataset for {self.config.source}")

    def stream(self) -> IterableDatasetDict | DataError:
        """Stream dataset from source"""
        if self.config is None:
             return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

        try:
            return load_dataset(
                path=self.config.source,
                name=None,
                split=None,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to download dataset {self.config.source}: {e}")
            return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

    def _tokenize_and_save(self, npy_path: Path) -> DataError | None:
        """Download dataset, tokenize it, and save as .npy file"""
        if self.config is None:
            return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

        try:
            # Load the dataset
            logger.info(f"Downloading dataset {self.config.source}")
            dataset = load_dataset(self.config.source, streaming=True)
            
            # Use configured tokenizer
            tokenizer_spec = self.config.tokenizer
            if not tokenizer_spec:
                 raise ValueError("Tokenizer must be specified in DatasetConfig")

            # Parse tokenizer string (format: "type:param" or just "type")
            tokenizer_parts = str(tokenizer_spec).split(":", 1)
            tokenizer_type = tokenizer_parts[0]
            
            tokenizer_cfg: TokenizerConfig
            
            if tokenizer_type == "tiktoken":
                if len(tokenizer_parts) < 2:
                     raise ValueError(f"Missing encoding for tiktoken tokenizer: {tokenizer_spec}")
                tokenizer_cfg = TiktokenTokenizerConfig(encoding=tokenizer_parts[1])
                
            elif tokenizer_type == "llama":
                if len(tokenizer_parts) < 2:
                     raise ValueError(f"Missing model_id for llama tokenizer: {tokenizer_spec}")
                tokenizer_cfg = LlamaTokenizerConfig(model_id=tokenizer_parts[1])
                
            elif tokenizer_type == "transformers":
                if len(tokenizer_parts) < 2:
                     raise ValueError(f"Missing model_id for transformers tokenizer: {tokenizer_spec}")
                # Re-use LlamaTokenizerConfig which maps to HuggingfaceTokenizer
                tokenizer_cfg = LlamaTokenizerConfig(type="llama", model_id=tokenizer_parts[1])
            else:
                 raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

            tokenizer = self._get_tokenizer_instance(tokenizer_cfg)

            # Collect and tokenize text
            tokens = []
            max_tokens = self.config.tokens if self.config.tokens else 1000000

            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    split = dataset[split_name]
                    for example in split:
                        if 'text' in example:
                            tokenized = tokenizer.encode(example['text'])
                            tokens.extend(tokenized)

                            if len(tokens) >= max_tokens:
                                break
                    if len(tokens) >= max_tokens:
                        break

            # Truncate
            tokens = tokens[:max_tokens]

            # Save
            if npy_path.parent.exists() is False:
                npy_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(npy_path, np.array(tokens, dtype=np.int32))
            logger.info(f"Saved tokenized dataset to {npy_path}")

            return None

        except Exception as e:
            logger.error(f"Failed to tokenize and save dataset {self.config.source}: {e}")
            return DataError(DataErrorType.DATASET_DOWNLOAD_FAILED)

    def _format_token_count(self, tokens: int) -> str:
        """Format token count to human readable string"""
        if tokens >= 1_000_000_000:
            return f"{int(tokens / 1_000_000_000)}b"
        if tokens >= 1_000_000:
            return f"{int(tokens / 1_000_000)}m"
        if tokens >= 1_000:
            return f"{int(tokens / 1_000)}k"
        return str(tokens)

    def _get_artifact_path(self) -> Path:
        """Get path to the tokenized dataset artifact"""
        if self.config is None:
            raise ValueError("Cannot get artifact path without config")

        # Handle simplified source names or full paths
        source_clean = self.config.source.strip("/")
        source_parts = source_clean.split("/")
        
        # Directory structure: artifacts/datasets/{source}
        # Example: artifacts/datasets/HuggingFaceFW/fineweb
        base_dir = Path("artifacts/datasets") / source_clean
        
        # Filename: {basename}_{tokens}.npy
        # Example: fineweb_100m.npy
        basename = source_parts[-1]
        tokens_str = self._format_token_count(self.config.tokens)
        filename = f"{basename}_{tokens_str}.npy"
        
        return base_dir / filename
