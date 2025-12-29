"""Tokenizer abstraction for evaluation.

Evaluation prompts need to be converted to token IDs. This module provides
a pluggable tokenizer interface with implementations for different backends.
Currently supports:
- tiktoken (used by GPT models)
- Hugging Face tokenizers (used by Llama-family models)
"""
from __future__ import annotations

import abc
import importlib
import importlib.util
from collections.abc import Callable, Sequence

from config.eval import LlamaTokenizerConfig, TiktokenTokenizerConfig, TokenizerConfig


class Tokenizer(abc.ABC):
    """Abstract base class for text-to-token encoding.

    Provides a consistent interface for evaluation code regardless of
    the underlying tokenizer implementation.
    """

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back to text."""


class _TiktokenTokenizer(Tokenizer):
    """Tiktoken-based tokenizer implementation.

    Wraps OpenAI's tiktoken library for efficient tokenization.
    Used for evaluating models with GPT-compatible tokenizers.
    """

    def __init__(self, *, encoding: str) -> None:
        """Initialize with a specific tiktoken encoding (e.g., 'cl100k_base')."""
        if not encoding:
            raise ValueError("encoding must be non-empty")
        if importlib.util.find_spec("tiktoken") is None:
            raise ImportError("tiktoken is required for tokenizer=tiktoken")
        mod = importlib.import_module("tiktoken")
        get_enc = getattr(mod, "get_encoding", None)
        if not callable(get_enc):
            raise ImportError("tiktoken.get_encoding is not available")
        self._enc = get_enc(str(encoding))

        # Cache and validate encode function
        encode_fn = getattr(self._enc, "encode", None)
        if not callable(encode_fn):
            raise ValueError("tiktoken encoding does not support encode(...)")
        self._encode_fn: Callable[[str], list[int]] = encode_fn  # type: ignore[assignment]

        # Cache and validate decode function
        decode_fn = getattr(self._enc, "decode", None)
        if not callable(decode_fn):
            raise ValueError("tiktoken encoding does not support decode(...)")
        self._decode_fn: Callable[[list[int]], str] = decode_fn  # type: ignore[assignment]

        # Smoke test
        test_ids = self._encode_fn("test")
        if not isinstance(test_ids, list) or (
            test_ids and not isinstance(test_ids[0], int)
        ):
            raise ValueError("tiktoken encode(...) does not return list[int]")

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs using tiktoken."""
        return self._encode_fn(str(text))

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs to text using tiktoken."""
        return str(self._decode_fn(list(ids)))


class _HFTokenizer(Tokenizer):
    """HuggingFace tokenizer implementation (e.g. LlamaTokenizerFast).

    We rely on `transformers`' AutoTokenizer so config only needs a model_id.
    """

    def __init__(self, *, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        try:
            mod = importlib.import_module("transformers")
        except Exception as e:
            raise ImportError("transformers is required for tokenizer=llama") from e
        auto = getattr(mod, "AutoTokenizer", None)
        if auto is None or not hasattr(auto, "from_pretrained"):
            raise ImportError("transformers.AutoTokenizer.from_pretrained is not available")

        # Keep defaults conservative: do not rely on remote code.
        self._tok = auto.from_pretrained(str(model_id), use_fast=True, trust_remote_code=False)

        encode = getattr(self._tok, "encode", None)
        decode = getattr(self._tok, "decode", None)
        if not callable(encode) or not callable(decode):
            raise ValueError("HuggingFace tokenizer must support encode(...) and decode(...)")

    def encode(self, text: str) -> list[int]:
        # For evaluation, we do not want BOS/EOS inserted implicitly.
        return list(self._tok.encode(str(text), add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """Build a Tokenizer from config.

    Factory function that creates the appropriate tokenizer based on
    the config type.
    """
    if isinstance(cfg, TiktokenTokenizerConfig):
        return _TiktokenTokenizer(encoding=str(cfg.encoding))
    if isinstance(cfg, LlamaTokenizerConfig):
        return _HFTokenizer(model_id=str(cfg.model_id))
    raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")
