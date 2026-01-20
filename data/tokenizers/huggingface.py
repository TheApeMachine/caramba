"""HuggingFace tokenizer implementation (e.g. LlamaTokenizerFast)

We rely on `transformers`' AutoTokenizer so config only needs a model_id.
"""
from __future__ import annotations

from transformers import AutoTokenizer
from typing import Sequence

from data.tokenizers.base import Tokenizer


class HuggingfaceTokenizer(Tokenizer):
    """HuggingFace tokenizer implementation (e.g. LlamaTokenizerFast).

    We rely on `transformers`' AutoTokenizer so config only needs a model_id.
    """

    def __init__(self, *, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        super().__init__(name=f"huggingface:{model_id}")
        # Keep defaults conservative: do not rely on remote code.
        self._tok = AutoTokenizer.from_pretrained(
            str(model_id),
            use_fast=True,
            trust_remote_code=False,
        )

    def encode(self, text: str) -> list[int]:
        # For evaluation, we do not want BOS/EOS inserted implicitly.
        return list(self._tok.encode(str(text), add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))
