"""Tiktoken-based tokenizer implementation

Wraps OpenAI's tiktoken library for efficient tokenization.
Used for evaluating models with GPT-compatible tokenizers.
"""
from __future__ import annotations

import tiktoken
from typing import Sequence

from caramba.data.tokenizers.base import Tokenizer

class TiktokenTokenizer(Tokenizer):
    """Tiktoken-based tokenizer implementation.

    Wraps OpenAI's tiktoken library for efficient tokenization.
    Used for evaluating models with GPT-compatible tokenizers.
    """

    def __init__(self, *, encoding: str) -> None:
        """Initialize with a specific tiktoken encoding (e.g., 'cl100k_base')."""
        if not encoding:
            raise ValueError("encoding must be non-empty")
        super().__init__(name=f"tiktoken:{encoding}")
        self._enc = tiktoken.get_encoding(str(encoding))

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs using tiktoken."""
        return list(self._enc.encode(str(text)))

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs to text using tiktoken."""
        return str(self._enc.decode(list(ids)))
