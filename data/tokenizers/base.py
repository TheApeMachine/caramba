"""Tokenizer base class

Provides a consistent interface for evaluation code regardless of
the underlying tokenizer implementation.
"""

from __future__ import annotations

import abc
from typing import Sequence


class Tokenizer(abc.ABC):
    """Abstract base class for text-to-token encoding.

    Provides a consistent interface for evaluation code regardless of
    the underlying tokenizer implementation.
    """
    def __init__(self, *, name: str) -> None:
        self.name = name

    @property
    def eos_token_id(self) -> int | None:
        """Best-effort EOS token id for generation stopping.

        Tokenizers that know their EOS/EOT id should override this. Returning
        None means "unknown", and generation should rely on max token limits.
        """
        return None

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back to text."""
