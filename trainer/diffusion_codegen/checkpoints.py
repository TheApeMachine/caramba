"""Checkpoint management

Handles saving and loading diffusion-codegen checkpoints, including optional
EMA state and manifest/config snapshots for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Checkpoint path conventions."""

    checkpoint_dir: Path
    run_id: str

    def stepPath(self, *, step: int) -> Path:
        return self.checkpoint_dir / f"{self.run_id}_step{int(step):08d}.pt"

    def latestSymlinkPath(self) -> Path:
        return self.checkpoint_dir / f"{self.run_id}_latest.pt"


@dataclass(frozen=True, slots=True)
class CheckpointManager:
    """Save/load checkpoints for diffusion codegen."""

    checkpoint_dir: str

    def ensureDir(self) -> Path:
        path = Path(self.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        *,
        run_id: str,
        step: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        scheduler_state: dict[str, Any] | None,
        ema_state: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> Path:
        """Save a checkpoint and update the latest pointer."""

        ckpt_dir = self.ensureDir()
        paths = CheckpointPaths(checkpoint_dir=ckpt_dir, run_id=str(run_id))
        ckpt = {
            "run_id": str(run_id),
            "step": int(step),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "ema_state": ema_state,
            "payload": payload,
        }
        out = paths.stepPath(step=int(step))
        torch.save(ckpt, out)
        torch.save(ckpt, paths.latestSymlinkPath())
        return out

    def load(self, *, checkpoint_path: str, map_location: torch.device) -> dict[str, Any]:
        """Load a checkpoint file."""

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=map_location)
        if not isinstance(ckpt, dict):
            raise TypeError(f"Checkpoint payload must be a dict, got {type(ckpt).__name__}")
        return ckpt

    def findLatest(self, *, run_id: str) -> Path:
        """Find the latest checkpoint for a run_id in checkpoint_dir."""

        ckpt_dir = self.ensureDir()
        latest = ckpt_dir / f"{str(run_id)}_latest.pt"
        if latest.exists():
            return latest

        candidates = sorted(ckpt_dir.glob(f"{str(run_id)}_step*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found for run_id={run_id!r} under {ckpt_dir}."
            )
        return candidates[-1]

