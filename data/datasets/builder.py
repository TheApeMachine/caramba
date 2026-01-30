"""Token dataset builder

Builds token datasets from path and block size. This keeps construction logic
out of training code so higher-level modules can simply orchestrate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data.npy import NpyDataset
from runtime.tensordict_utils import TensorDictBase
from torch.utils.data import Dataset


class TokenDatasetBuilder:
    """Build token datasets from path and block size."""

    @staticmethod
    def build(
        *,
        path: Path | str,
        block_size: int,
        append_eos: bool = False,
        tokenizer: Any | None = None,
    ) -> Dataset[TensorDictBase]:
        """Build a token dataset from a NumPy file path.

        Args:
            path: Path to the `.npy` file containing tokenized data
            block_size: Sequence length for each training sample
            append_eos: Whether to force an EOS token at the end of each target block.
            tokenizer: Tokenizer instance (required if append_eos=True)

        Returns:
            A dataset instance that serves (input, target) token pairs
        """
        eos_id: int | None = None
        if append_eos:
            if tokenizer is None:
                raise ValueError("append_eos=True requires a tokenizer to resolve EOS ID")
            # Try common attributes for EOS ID
            if hasattr(tokenizer, "eos_token_id") and getattr(tokenizer, "eos_token_id") is not None:
                eos_id = int(getattr(tokenizer, "eos_token_id"))
            elif hasattr(tokenizer, "eos_id") and getattr(tokenizer, "eos_id") is not None:
                eos_id = int(getattr(tokenizer, "eos_id"))
            else:
                raise ValueError(f"Could not determine eos_token_id from tokenizer {type(tokenizer)}")

        return NpyDataset(
            str(path),
            block_size=int(block_size),
            append_eos=append_eos,
            eos_id=eos_id,
        )
