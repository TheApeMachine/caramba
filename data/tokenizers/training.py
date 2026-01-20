"""Training tokenizer interfaces.

Caramba has two tokenizer "worlds":

- Eval tokenizers: small interface used by eval/benchmarks (see `base.Tokenizer`)
- Training tokenizers: used by datasets and diffusion-codegen utilities, which
  need `token_to_id`, `encode(...).ids`, and `decode(..., skip_special_tokens=...)`.

This module defines a small Protocol so training code can depend on Caramba's
interface instead of importing third-party tokenizer types directly.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from data.tokenizers.encoding import Encoding

class TrainingTokenizer(Protocol):
    """Protocol for training-grade tokenizers used in datasets/training code."""

    def encode(self, text: str) -> Encoding:
        """Encode text into token IDs."""
        ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...

    def token_to_id(self, token: str) -> int | None:
        """Resolve a token string to its integer ID, if present."""
        ...

