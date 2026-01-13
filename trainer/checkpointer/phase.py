"""Phase-based checkpointer implementation."""
from __future__ import annotations

from pathlib import Path
import torch

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.console import logger
from caramba.trainer.context.run import RunCtx
from caramba.trainer.checkpointer.base import CheckPointer


class PhaseCheckPointer(CheckPointer):
    """Checkpointer implementation for phase-based training."""
    def __init__(self, *, ctx: RunCtx, manifest: Manifest, target: Target) -> None:
        """Create a phase checkpointer.

        The phase checkpointer saves the training state at predetermined
        step intervals during phase-based training progression.
        """
        super().__init__(ctx=ctx, manifest=manifest, target=target)
        self.manifest = manifest
        self.target = target
        self.ctx = ctx

    def step(self) -> None:
        """Save a checkpoint and return its path."""
        filename_parts = [
            f"run{self.ctx.run_id}",
            f"phase{self.ctx.phase}",
            f"step{self.ctx.step}"
        ]

        filename = "_".join(filename_parts) + ".pt"
        torch.save(obj=self.ctx.model_dump(), f=filename)

        logger.info(f"Checkpoint saved: {filename}")

    def load_resume(self, *, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=self.ctx.device, weights_only=False)

        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint must be a dict, got {type(state).__name__}")

        self.ctx.student.load_state_dict(state["student_state_dict"])

        logger.success(
            f"Resumed from checkpoint: run={state.get('run_id')}, "
            f"phase={state.get('phase')}, step={state.get('step')}"
        )
