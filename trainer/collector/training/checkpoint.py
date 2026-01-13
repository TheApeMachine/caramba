"""Checkpoint hooks for training."""

from __future__ import annotations

from pathlib import Path

import torch

from caramba.collector.training import TrainHook


class FinalCheckpointHook(TrainHook):
    """Save a single final checkpoint at the end of a run."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        run_id: str,
        phase: str,
        system: object,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._run_id = str(run_id)
        self._phase = str(phase)
        self._system = system

    def on_run_end(self, *, step: int) -> None:
        filename = f"{self._run_id}_{self._phase}_final.pt"
        path = self._checkpoint_dir / filename
        if not hasattr(self._system, "state_dict"):
            raise TypeError("System component does not expose state_dict()")
        state = {
            "system_state_dict": self._system.state_dict(),  # type: ignore[attr-defined]
            "run_id": self._run_id,
            "step": int(step),
        }
        torch.save(state, path)

