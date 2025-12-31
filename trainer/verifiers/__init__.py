"""Verifier components for Upcycle.

These are small, swappable objects instantiated via `Config.build()` (same
pattern as layers/topologies).
"""

from __future__ import annotations

from typing import Protocol

from caramba.config.run import Run

from caramba.trainer.upcycle_context import UpcycleContext


class Verifier(Protocol):
    def verify(self, run: Run, ctx: UpcycleContext) -> None: ...


__all__ = ["Verifier"]

