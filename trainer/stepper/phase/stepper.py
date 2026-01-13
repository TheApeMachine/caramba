"""Phase-based stepper implementation

`PhaseStepper` is a thin orchestrator around the global fine-tuning loop used
in phase-based training progression (blockwise → global phases).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.stepper.base import Stepper
from caramba.trainer.stepper.phase.loop import PhaseLoop

if TYPE_CHECKING:
    from caramba.config.run import Run
    from caramba.trainer.collector.base import Collector
    from caramba.trainer.checkpointer.base import CheckPointer
    from caramba.trainer.upcycle_context import UpcycleContext


class PhaseStepper(Stepper):
    """Phase-based fine-tuning stepper

    Implements the global fine-tuning loop used after blockwise distillation
    in the upcycling workflow.
    """

    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        """Create a phase stepper bound to a manifest target."""
        super().__init__(manifest=manifest, target=target)
        self.loop = PhaseLoop()

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
        """Execute the global fine-tuning loop for a single run."""
        self.loop.run(
            run=run,
            ctx=ctx,
            collector=collector,
            checkpointer=checkpointer,
            save_every=int(save_every),
            resume_state=resume_state,
        )

