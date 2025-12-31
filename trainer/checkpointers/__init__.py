"""Checkpointer components for training sessions."""

from __future__ import annotations

from typing import Protocol

from pathlib import Path

from torch.optim import Optimizer

from caramba.trainer.upcycle_context import UpcycleContext


class CheckPointer(Protocol):
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
    ) -> Path: ...

    def load_resume(self, *, ctx: UpcycleContext, path: Path) -> dict[str, object]: ...

    def latest(self, *, ctx: UpcycleContext, run_id: str, phase: str) -> Path | None: ...


__all__ = ["CheckPointer"]

