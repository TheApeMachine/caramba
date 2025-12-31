from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from caramba.config.eval import LlamaTokenizerConfig, TiktokenTokenizerConfig
from caramba.eval.tokenizer import build_tokenizer


def test_build_tokenizer_tiktoken() -> None:
    tok = build_tokenizer(TiktokenTokenizerConfig(encoding="cl100k_base"))
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids)


def test_build_tokenizer_llama_uses_transformers_autotokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide a tiny fake `transformers` module to avoid network/filesystem dependencies.
    fake = types.ModuleType("transformers")

    class _DummyTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False):
            assert add_special_tokens is False
            return [1, 2, 3]

        def decode(self, ids, *, skip_special_tokens: bool = True):
            assert skip_special_tokens is True
            return "ok"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            assert model_id
            assert kwargs.get("use_fast") is True
            assert kwargs.get("trust_remote_code") is False
            return _DummyTokenizer()

    fake_any: Any = fake
    fake_any.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    tok = build_tokenizer(LlamaTokenizerConfig(model_id="meta-llama/Llama-3.2-1B"))
    assert tok.encode("hi") == [1, 2, 3]
    assert tok.decode([1, 2, 3]) == "ok"

