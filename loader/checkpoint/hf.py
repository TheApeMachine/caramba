"""Hugging Face Hub integration for downloading model checkpoints.

Instead of manually downloading and managing checkpoint files, you can
reference models by their Hub ID (e.g., "meta-llama/Llama-3.2-1B") and
this module handles the download, caching, and path resolution.
"""
from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download

from core.platform import Platform
from loader.checkpoint.base import Checkpoint
from loader.checkpoint.error import CheckpointError, CheckpointErrorType


class HFCheckpoint(Checkpoint):
    """Downloads model files from Hugging Face Hub.

    Handles both simple downloads (just the repo ID) and specific file
    paths within a repo. Files are cached locally so subsequent loads
    don't require re-downloading.
    """
    def __init__(
        self,
        repo_id: str,
        platform: Platform = Platform(),
    ) -> None:
        """Initialize the loader for a specific repo."""
        super().__init__(platform)

        if not repo_id:
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_NOT_FOUND)

        self.repo_id = repo_id

    def load(self) -> Path:
        """Download the model"""
        return Path(
            snapshot_download(
                repo_id=self.repo_id,
                ignore_patterns=["*.msgpack", "*.h5"],
                local_dir=self.base_path,
            )
        )
