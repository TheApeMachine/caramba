"""Code BPE tokenizer

Provides a deterministic, manifest-configured BPE tokenizer for source code
corpora. This exists so experiments can reproduce tokenization exactly without
hidden environment configuration or ad-hoc scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer


@dataclass(frozen=True, slots=True)
class CodeBpeTokenizer:
    """BPE tokenizer for code

    This tokenizer is trained on source code files and persists to a single
    JSON file. It uses Metaspace pre-tokenization/decoding to make whitespace
    explicit and stable across languages and editors.
    """

    vocab_size: int
    special_tokens: list[str]
    file_extensions: list[str]

    def train(self, *, data_dir: str, output_file: str) -> Path:
        """Train and save a tokenizer from `data_dir`.

        Raises if no matching files exist, or if training fails.
        """

        root = Path(data_dir)
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(
                f"Tokenizer training data_dir must be a directory: {root}"
            )
        if not root.exists():
            raise FileNotFoundError(
                f"Tokenizer training data_dir does not exist: {root}. "
                "Set trainer.config.tokenizer.data_dir to a valid directory."
            )

        files = self.listFiles(root=root)
        if not files:
            raise ValueError(
                "No source code files found for tokenizer training. "
                f"data_dir={root}, extensions={self.file_extensions}"
            )

        iterator = self.makeTextIterator(files=files)
        tokenizer = self.makeTokenizer()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(files))

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out_path))
        return out_path

    def load(self, *, tokenizer_file: str) -> HFTokenizer:
        """Load a previously saved tokenizer.

        Raises if the tokenizer file is missing or malformed.
        """

        path = Path(tokenizer_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {path}. "
                "Train it first or set trainer.config.tokenizer.tokenizer_file correctly."
            )
        tokenizer = HFTokenizer.from_file(str(path))
        self.validateSpecialTokens(tokenizer=tokenizer)
        return tokenizer

    def validateSpecialTokens(self, *, tokenizer: HFTokenizer) -> None:
        """Validate required special tokens exist.

        This keeps downstream training/generation errors actionable.
        """

        missing: list[str] = []
        for token in self.special_tokens:
            if tokenizer.token_to_id(token) is None:
                missing.append(token)
        if missing:
            raise ValueError(
                "Tokenizer is missing required special tokens. "
                f"missing={missing}, expected={self.special_tokens}"
            )

    def listFiles(self, *, root: Path) -> list[Path]:
        """List files matching configured extensions.

        Uses deterministic ordering for reproducibility.
        """

        if not self.file_extensions:
            raise ValueError("file_extensions must be non-empty.")
        # Normalize extensions to match Path.suffix (leading dot) and compare
        # case-insensitively. This avoids mismatches like "py" vs p.suffix ".py".
        exts: set[str] = set()
        for e in self.file_extensions:
            if not isinstance(e, str) or not e.strip():
                raise ValueError(
                    "file_extensions entries must be non-empty strings. "
                    f"Got invalid entry: {e!r}"
                )
            ext = e.strip().lower()
            if not ext.startswith("."):
                ext = "." + ext
            exts.add(ext)
        if not exts:
            raise ValueError("file_extensions must be non-empty.")

        files: list[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in exts:
                files.append(p)
        files.sort(key=lambda x: str(x))
        return files

    def makeTokenizer(self) -> HFTokenizer:
        """Create the Tokenizers tokenizer instance."""

        tokenizer = HFTokenizer(BPE(unk_token=self.unkToken()))
        # tokenizers exposes these as dynamic properties; stubs are overly strict.
        tokenizer.pre_tokenizer = Metaspace()  # type: ignore[assignment]
        tokenizer.decoder = MetaspaceDecoder()  # type: ignore[assignment]
        return tokenizer

    def makeTextIterator(self, *, files: list[Path]) -> Iterable[str]:
        """Create a streaming iterator over file text."""

        for p in files:
            try:
                yield p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                raise RuntimeError(f"Failed to read tokenizer training file: {p}") from e

    def unkToken(self) -> str:
        """Select the unknown token."""

        if "<unk>" in self.special_tokens:
            return "<unk>"
        raise ValueError(
            "special_tokens must include '<unk>' so the tokenizer has a defined unknown token."
        )

