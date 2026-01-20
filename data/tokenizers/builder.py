"""Tokenizer builder

Builds tokenizers from config objects. This keeps construction logic out of
benchmarks/eval code so higher-level modules can simply orchestrate.
"""
from __future__ import annotations

from config.eval import (
    CodeBpeTokenizerConfig,
    LlamaTokenizerConfig,
    TiktokenTokenizerConfig,
    TokenizerConfig,
)
from data.tokenizers.base import Tokenizer
from data.tokenizers.bpe_eval import CodeBpeEvalTokenizer
from data.tokenizers.huggingface import HuggingfaceTokenizer
from data.tokenizers.tiktoken import TiktokenTokenizer


class TokenizerBuilder:
    """Build tokenizers from `caramba.config.eval.TokenizerConfig`."""

    def build(self, cfg: TokenizerConfig) -> Tokenizer:
        if isinstance(cfg, TiktokenTokenizerConfig):
            return TiktokenTokenizer(encoding=str(cfg.encoding))
        if isinstance(cfg, LlamaTokenizerConfig):
            return HuggingfaceTokenizer(model_id=str(cfg.model_id))
        if isinstance(cfg, CodeBpeTokenizerConfig):
            return CodeBpeEvalTokenizer(tokenizer_file=str(cfg.tokenizer_file))
        raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")

