"""Code chunk dataset

Builds a prompt/target paired dataset from a source tree:
- target_ids: a token window from a file
- prompt_ids: the preceding context window from the same file

This teaches conditional generation where the model learns to continue code from
previous context, while still supporting unconditional generation via CFG by
masking the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


class CodeChunksTorchDataset(Dataset[TensorDictBase]):
    """Torch dataset yielding paired (target_ids, prompt_ids) windows.

    This class is deterministic: file ordering and chunk indexing are stable so
    that runs are reproducible when seeded.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        tokenizer: Tokenizer,
        seq_len: int,
        stride: int,
        extensions: list[str],
        max_files: int | None,
        cache_size: int,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.extensions = [e.lower() for e in extensions]
        self.max_files = max_files
        self.cache_size = int(cache_size)

        self.padId = self.requirePadId()
        self.files = self.listFiles()
        self.cache: dict[str, list[int]] = {}
        self.chunks = self.buildChunks()

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> TensorDictBase:
        path, target_start, target_end = self.chunks[int(idx)]
        target_ids = self.loadTokenSlice(path=path, start=target_start, end=target_end)

        prompt_end = int(target_start) if int(target_start) > 0 else int(target_end)
        prompt_start = max(0, int(prompt_end) - int(self.seq_len))
        prompt_ids = self.loadTokenSlice(path=path, start=prompt_start, end=prompt_end)

        target = self.padOrTruncate(ids=target_ids)
        prompt = self.padOrTruncate(ids=prompt_ids)

        return as_tensordict(
            {
                "target_ids": torch.tensor(target, dtype=torch.long),
                "prompt_ids": torch.tensor(prompt, dtype=torch.long),
            }
        )

    def requirePadId(self) -> int:
        """Require that the tokenizer defines <pad>."""

        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer must define a <pad> token.")
        return int(pad_id)

    def listFiles(self) -> list[Path]:
        """List code files deterministically."""

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset data_dir does not exist: {self.data_dir}. "
                "Set data.config.data_dir to a valid directory."
            )
        exts = set(self.extensions)
        if not exts:
            raise ValueError("extensions must be non-empty.")

        files: list[Path] = []
        for p in self.data_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in exts:
                files.append(p)
        files.sort(key=lambda x: str(x))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise ValueError(
                f"No files found under {self.data_dir} with extensions={sorted(exts)}"
            )
        return files

    def buildChunks(self) -> list[tuple[str, int, int]]:
        """Build (file, start, end) token ranges."""

        chunks: list[tuple[str, int, int]] = []
        min_len = max(1, int(self.seq_len) // 2)

        for p in self.files:
            ids = self.loadAllTokens(path=str(p))
            if not ids:
                raise RuntimeError(
                    f"Tokenizer produced empty ids for file: {p}. "
                    "Ensure the file is readable and contains text."
                )
            for start in range(0, len(ids), int(self.stride)):
                end = min(start + int(self.seq_len), len(ids))
                if (end - start) >= min_len:
                    chunks.append((str(p), int(start), int(end)))

        if not chunks:
            raise RuntimeError(
                "No valid chunks produced. "
                f"seq_len={self.seq_len}, stride={self.stride}, files={len(self.files)}"
            )
        return chunks

    def loadTokenSlice(self, *, path: str, start: int, end: int) -> list[int]:
        """Load a token slice, using a bounded per-process cache."""

        ids = self.loadAllTokens(path=path)
        return ids[int(start) : int(end)]

    def loadAllTokens(self, *, path: str) -> list[int]:
        """Load all token IDs for a file, caching by path."""

        if path in self.cache:
            return self.cache[path]

        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read dataset file: {p}. "
                "Fix file permissions or exclude the path from the dataset."
            ) from e

        ids = self.tokenizer.encode(text).ids
        if len(self.cache) >= int(self.cache_size):
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = list(ids)
        return self.cache[path]

    def padOrTruncate(self, *, ids: list[int]) -> list[int]:
        """Pad or truncate to seq_len."""

        if len(ids) >= int(self.seq_len):
            return ids[: int(self.seq_len)]
        return ids + [int(self.padId)] * (int(self.seq_len) - len(ids))


@dataclass(frozen=True, slots=True)
class CodeChunksDataset:
    """Code chunk dataset component

    This is a dataset component (manifest `data.ref`) that can be built into a
    torch Dataset yielding TensorDict batches.
    """

    data_dir: str
    tokenizer_file: str
    seq_len: int = 128
    stride: int | None = None
    extensions: list[str] | None = None
    max_files: int | None = None
    cache_size: int = 200

    def build(self) -> Dataset[TensorDictBase]:
        """Build the torch Dataset."""

        tokenizer = Tokenizer.from_file(str(Path(self.tokenizer_file)))
        stride = int(self.stride) if self.stride is not None else max(1, int(self.seq_len) // 2)
        extensions = self.extensions or [
            ".py",
            ".js",
            ".ts",
            ".go",
            ".java",
            ".cs",
            ".cpp",
            ".c",
        ]

        return CodeChunksTorchDataset(
            data_dir=Path(self.data_dir),
            tokenizer=tokenizer,
            seq_len=int(self.seq_len),
            stride=int(stride),
            extensions=list(extensions),
            max_files=self.max_files,
            cache_size=int(self.cache_size),
        )

    def config(self) -> dict[str, Any]:
        """Return a serializable config payload (for checkpoints)."""

        return {
            "data_dir": str(self.data_dir),
            "tokenizer_file": str(self.tokenizer_file),
            "seq_len": int(self.seq_len),
            "stride": int(self.stride) if self.stride is not None else None,
            "extensions": list(self.extensions or []),
            "max_files": int(self.max_files) if self.max_files is not None else None,
            "cache_size": int(self.cache_size),
        }

