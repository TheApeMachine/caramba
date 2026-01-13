"""Stepper base class

The stepper base class is the base class for all steppers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

from caramba.manifest import Manifest
from caramba.manifest.target import Target

if TYPE_CHECKING:
    from caramba.config.run import Run
    from caramba.trainer.collector.base import Collector
    from caramba.trainer.checkpointer.base import CheckPointer
    from caramba.trainer.upcycle_context import UpcycleContext


class Stepper(ABC):
    """Training stepper

    A stepper owns a single training loop (or a small family of closely related
    loops) and is responsible for orchestrating the pieces that implement it.
    """

    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        """Create a stepper bound to a manifest target."""
        self.manifest = manifest
        self.target = target

    @abstractmethod
    def run(
        self,
        run: "Run",
        ctx: "UpcycleContext",
        *,
        collector: "Collector",
        checkpointer: "CheckPointer",
        save_every: int,
        resume_state: dict[str, object] | None,
    ) -> None:
        """Execute the stepper for a single run."""
        raise NotImplementedError("Subclasses must implement run()")
