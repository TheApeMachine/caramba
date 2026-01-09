"""Code chunk dataset

Loads code files and creates (prompt, target) pairs where target is a token
window and prompt is the preceding context. This enables conditional code
generation training while supporting unconditional generation through prompt
masking in classifier-free guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from caramba.data.tokenizers.training import TrainingTokenizer
from caramba.data.tokenizers.hf_json import HfJsonTokenizer
from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


class CodeChunksTorchDataset(Dataset[TensorDictBase]):
    """Code chunks dataset implementation

    Processes code files into overlapping windows with prompt/target pairs,
    caching tokenized files to avoid repeated tokenization. File ordering and
    chunking are deterministic for reproducible training runs.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        tokenizer: TrainingTokenizer,
        seq_len: int,
        stride: int,
        extensions: list[str],
        max_files: int | None,
        cache_size: int,
    ) -> None:
        """Initialize code chunks dataset

        Scans the data directory for code files, builds a chunk index, and sets
        up a bounded cache to store tokenized file contents for efficient
        repeated access during training.
        """
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
        """Get dataset length

        Returns the number of chunks across all files, which depends on file
        sizes, sequence length, and stride parameters.
        """
        return len(self.chunks)

    def __getitem__(self, idx: int) -> TensorDictBase:
        """Get code chunk sample

        Extracts a target window and its preceding prompt context from the
        same file, then pads or truncates both to the configured sequence
        length for consistent batch shapes.
        """
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
        """Get padding token ID

        Ensures the tokenizer has a padding token defined, which is required
        for creating fixed-length sequences from variable-length code chunks.
        """
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer must define a <pad> token.")
        return int(pad_id)

    def listFiles(self) -> list[Path]:
        """List code files

        Recursively scans the data directory for files matching the configured
        extensions, sorting them deterministically for reproducible dataset
        ordering across runs.
        """
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
        """Build chunk index

        Creates (file, start, end) tuples representing all valid token windows
        across all files, using the configured stride to control overlap
        between chunks.
        """
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
        """Load token slice

        Extracts a contiguous range of tokens from a file, using the cached
        tokenized content to avoid re-tokenizing on every access.
        """
        ids = self.loadAllTokens(path=path)
        return ids[int(start) : int(end)]

    def loadAllTokens(self, *, path: str) -> list[int]:
        """Load and cache file tokens

        Tokenizes a file and stores the result in a bounded cache, evicting
        the oldest entry when the cache is full. This balances memory usage
        with access speed for frequently accessed files.
        """
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
        """Pad or truncate sequence

        Ensures sequences are exactly seq_len tokens by truncating longer
        sequences or padding shorter ones with the padding token, maintaining
        consistent batch shapes for efficient training.
        """
        if len(ids) >= int(self.seq_len):
            return ids[: int(self.seq_len)]
        return ids + [int(self.padId)] * (int(self.seq_len) - len(ids))


@dataclass(frozen=True, slots=True)
class CodeChunksDataset:
    """Code chunks dataset component

    Manifest-level dataset that processes code files into prompt/target pairs
    for conditional generation training. Supports multiple programming languages
    and configurable chunking strategies.
    """

    data_dir: str
    tokenizer_file: str
    seq_len: int = 128
    stride: int | None = None
    extensions: list[str] | None = None
    max_files: int | None = None
    cache_size: int = 200

    def build(self) -> Dataset[TensorDictBase]:
        """Build code chunks dataset

        Loads the tokenizer, sets default file extensions if not specified,
        and creates the PyTorch dataset that will process code files into
        training samples.
        """
        tokenizer = HfJsonTokenizer.from_file(tokenizer_file=str(Path(self.tokenizer_file)))
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
        """Get serializable config

        Returns a dictionary representation suitable for saving in checkpoints
        or logging, ensuring all parameters can be reconstructed later.
        """
        return {
            "data_dir": str(self.data_dir),
            "tokenizer_file": str(self.tokenizer_file),
            "seq_len": int(self.seq_len),
            "stride": int(self.stride) if self.stride is not None else None,
            "extensions": list(self.extensions or []),
            "max_files": int(self.max_files) if self.max_files is not None else None,
            "cache_size": int(self.cache_size),
        }

