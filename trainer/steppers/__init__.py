"""Stepper components (training-loop orchestrators)."""

from __future__ import annotations

from typing import Protocol

from config.run import Run

from trainer.collectors import Collector
from trainer.checkpointers import CheckPointer
from trainer.upcycle_context import UpcycleContext


class Stepper(Protocol):
    def run(
        self,
        run: Run,
        ctx: UpcycleContext,
        *,
        collector: Collector,
        checkpointer: CheckPointer,
        save_every: int,
        resume_state: dict[str, object] | None,
    ) -> None: ...


__all__ = ["Stepper"]

