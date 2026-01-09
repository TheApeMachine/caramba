"""Tokenizer loaded from a HuggingFace `tokenizers` JSON file.

This is the primary training-time tokenizer representation in Caramba: datasets
and diffusion-codegen utilities consume a small interface (`TrainingTokenizer`)
and do not import `tokenizers` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer as HFTokenizer

from caramba.data.tokenizers.encoding import Encoding
from caramba.data.tokenizers.training import TrainingTokenizer


@dataclass(frozen=True, slots=True)
class HfJsonTokenizer(TrainingTokenizer):
    """Wrap a `tokenizers.Tokenizer` loaded from JSON."""

    name: str
    _tok: HFTokenizer

    @classmethod
    def from_file(cls, *, tokenizer_file: str, name: str | None = None) -> "HfJsonTokenizer":
        path = Path(str(tokenizer_file))
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        tok = HFTokenizer.from_file(str(path))
        tok_name = str(name) if name is not None else f"hf_json:{path}"
        return cls(name=tok_name, _tok=tok)

    def encode(self, text: str) -> Encoding:
        # The underlying HF tokenizer returns an Encoding-like object; we copy
        # ids into a small stable dataclass to avoid leaking third-party types.
        enc = self._tok.encode(str(text))
        return Encoding(ids=list(getattr(enc, "ids")))

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return str(
            self._tok.decode(list(ids), skip_special_tokens=bool(skip_special_tokens))
        )

    def token_to_id(self, token: str) -> int | None:
        out = self._tok.token_to_id(str(token))
        return None if out is None else int(out)

