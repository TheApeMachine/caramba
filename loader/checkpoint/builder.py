"""Checkpoint builder.

Selects an appropriate checkpoint loader implementation based on the path.
This keeps format detection centralized and manifest-driven.
"""

from __future__ import annotations

from pathlib import Path

from core.platform import Platform
from loader.checkpoint.base import Checkpoint
from loader.checkpoint.base import StateDict
from loader.checkpoint.pytorch import CheckpointPytorch
from loader.checkpoint.safetensors import CheckpointSafetensors
from loader.checkpoint.sharded import CheckpointSharded


class CheckpointBuilder:
    """Build checkpoint loaders from a path."""

    @staticmethod
    def _resolve_file_in_dir(dir_path: Path) -> Path:
        """Resolve a concrete checkpoint file from a directory.

        HuggingFace `snapshot_download()` returns a directory. We treat that
        directory as a checkpoint "container" and select an appropriate file
        inside it (safetensors, sharded index, or a pytorch .pt/.bin/.pth file).
        """
        p = Path(dir_path)
        if not p.is_dir():
            return p

        # Common HF filenames (prefer sharded indices, then single-file safetensors, then pytorch).
        direct_candidates = [
            p / "model.safetensors.index.json",
            p / "model.safetensors",
            p / "pytorch_model.bin.index.json",
            p / "pytorch_model.bin",
            p / "model.pt",
            p / "model.bin",
            p / "consolidated.00.pth",
            p / "original" / "consolidated.00.pth",
        ]
        for c in direct_candidates:
            if c.exists() and c.is_file():
                return c

        # Generic fallback search (recursive; first match wins).
        patterns = [
            "**/*.safetensors.index.json",
            "**/*.index.json",
            "**/*.safetensors",
            "**/*.bin",
            "**/*.pth",
            "**/*.pt",
        ]
        for pat in patterns:
            for c in p.glob(pat):
                if c.is_file():
                    return c

        # Nothing usable found; return the original directory so loaders can raise
        # a more specific error if desired.
        return p

    def build(self, path: Path, platform: Platform = Platform()) -> Checkpoint:
        p = Path(path)

        if p.name.endswith(".index.json"):
            return CheckpointSharded(builder=self)

        if p.suffix == ".safetensors":
            return CheckpointSafetensors(platform)

        return CheckpointPytorch(platform)

    def load(self, path: Path) -> StateDict:
        """Load a checkpoint as a state_dict."""
        p = Path(path)
        if p.is_dir():
            resolved = self._resolve_file_in_dir(p)
            # If resolution failed, fall through so the loader can error.
            if resolved != p:
                return self.load(resolved)
        return self.build(p).load(p)

