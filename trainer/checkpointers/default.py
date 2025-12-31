"""Default checkpointer implementation (Upcycle session)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer

from caramba.config.checkpointer import DefaultCheckPointerConfig
from caramba.console import logger
from caramba.trainer.upcycle_context import UpcycleContext


class DefaultCheckPointer:
    def __init__(self, config: DefaultCheckPointerConfig) -> None:
        self.config = config

    def save(
        self,
        *,
        ctx: UpcycleContext,
        run_id: str,
        phase: str,
        step: int,
        block_index: int | None = None,
        block_step: int | None = None,
        global_step: int | None = None,
        optimizer: Optimizer | None = None,
        scheduler: object | None = None,
        is_final: bool = False,
    ) -> Path:
        if is_final:
            filename = f"{run_id}_{phase}_final.pt"
        elif block_index is not None:
            filename = f"{run_id}_{phase}_block{block_index}_step{step}.pt"
        else:
            filename = f"{run_id}_{phase}_step{step}.pt"

        path = ctx.checkpoint_dir / filename

        state: dict[str, object] = {
            "student_state_dict": ctx.student.state_dict(),
            "run_id": str(run_id),
            "phase": str(phase),
            "step": int(step),
            "block_index": block_index,
            "block_step": block_step,
            "global_step": int(global_step if global_step is not None else step),
        }
        if optimizer is not None:
            try:
                state["optimizer_state_dict"] = optimizer.state_dict()
            except Exception:
                pass
        if scheduler is not None:
            try:
                state["scheduler_state_dict"] = getattr(scheduler, "state_dict")()
            except Exception:
                pass

        if ctx.dist_ctx is not None:
            ctx.dist_ctx.save_checkpoint(state, str(path))
        else:
            torch.save(state, path)

        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_resume(self, *, ctx: UpcycleContext, path: Path) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=ctx.device, weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint must be a dict, got {type(state).__name__}")
        self._validate(state)

        ctx.student.load_state_dict(state["student_state_dict"])  # type: ignore[arg-type]
        logger.success(
            f"Resumed from checkpoint: run={state.get('run_id')}, "
            f"phase={state.get('phase')}, step={state.get('step')}"
        )
        return state

    def latest(self, *, ctx: UpcycleContext, run_id: str, phase: str) -> Path | None:
        final_path = ctx.checkpoint_dir / f"{run_id}_{phase}_final.pt"
        if final_path.exists():
            return final_path

        pattern = f"{run_id}_{phase}_*.pt"
        checkpoints = list(ctx.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    @staticmethod
    def _validate(state: dict[str, object]) -> None:
        required = ["student_state_dict", "run_id", "phase", "step"]
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")
        if not isinstance(state.get("student_state_dict"), dict):
            raise TypeError("Checkpoint student_state_dict must be a dict")
        _ = str(state.get("run_id"))
        _ = str(state.get("phase"))
        try:
            int(state.get("step"))  # type: ignore[arg-type]
        except Exception as e:
            raise TypeError("Checkpoint step must be int-like") from e

