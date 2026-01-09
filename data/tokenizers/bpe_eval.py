"""Evaluation wrapper for a CodeBPE tokenizer JSON.

`CodeBpeTokenizer` (in `bpe.py`) is responsible for training/saving/loading the
underlying HuggingFace `tokenizers` JSON. This module provides an eval-facing
wrapper that matches Caramba's small eval tokenizer interface (`base.Tokenizer`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer as HFTokenizer

from caramba.data.tokenizers.base import Tokenizer


class CodeBpeEvalTokenizer(Tokenizer):
    """Eval tokenizer backed by a HuggingFace `tokenizers` JSON file."""

    def __init__(self, *, tokenizer_file: str) -> None:
        path = Path(str(tokenizer_file))
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        super().__init__(name=f"code_bpe:{path}")

        self._tok = HFTokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(str(text)).ids)

    def decode(self, ids: Sequence[int]) -> str:
        # Follow the same convention as HuggingFace eval tokenizer:
        # skip special tokens by default.
        return str(self._tok.decode(list(ids), skip_special_tokens=True))

