"""Tokenizer operations for diffusion codegen

Keeps tokenizer training/loading policy explicit and manifest-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.data.tokenizers.bpe import CodeBpeTokenizer


@dataclass(frozen=True, slots=True)
class TokenizerManager:
    """Tokenizer file manager

    Ensures a tokenizer file exists. Training is only performed when explicitly
    enabled in config.
    """

    config: dict[str, Any] | None

    def ensureTokenizerFile(self) -> str:
        cfg = self.config or {}
        tokenizer_file = str(cfg.get("tokenizer_file", "runs/tokenizer/bpe_tokenizer.json"))
        if Path(tokenizer_file).exists():
            return tokenizer_file

        if not bool(cfg.get("train_if_missing", False)):
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_file}. "
                "Set trainer.config.tokenizer.train_if_missing=true to allow training."
            )
        return str(self.trainTokenizer(tokenizer_file=tokenizer_file, cfg=cfg))

    def trainTokenizer(self, *, tokenizer_file: str, cfg: dict[str, Any]) -> Path:
        data_dir = str(cfg.get("data_dir", "."))
        vocab_size = int(cfg.get("vocab_size", 50_000))
        special_tokens = list(cfg.get("special_tokens", ["<unk>", "<pad>", "<s>", "</s>"]))
        extensions = list(
            cfg.get("file_extensions", [".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c"])
        )
        tokenizer = CodeBpeTokenizer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            file_extensions=extensions,
        )
        return tokenizer.train(data_dir=data_dir, output_file=str(tokenizer_file))

