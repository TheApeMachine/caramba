"""Phase dispatcher stepper

Dispatches training execution to phase-specific steppers based on the
configured training phase (blockwise, global, orchestrated).
"""

from __future__ import annotations

from caramba.config.train import TrainPhase
from caramba.trainer.collectors import Collector
from caramba.trainer.checkpointers import CheckPointer
from caramba.trainer.steppers.blockwise import BlockwiseStepper
from caramba.trainer.steppers.global_stepper import GlobalStepper
from caramba.trainer.steppers.global_orchestrated import GlobalOrchestratedStepper
from caramba.trainer.upcycle_context import UpcycleContext
from caramba.config.run import Run
from caramba.config.stepper import PhaseDispatcherConfig


class PhaseDispatcherStepper:
    """Phase dispatcher stepper

    Routes training execution to the appropriate phase-specific stepper
    (blockwise, global, or orchestrated) based on manifest configuration.
    """

    def __init__(self, config: PhaseDispatcherConfig) -> None:
        self.config = config
        self.blockwise = BlockwiseStepper()
        self.global_stepper = GlobalStepper()
        self.orchestrated = GlobalOrchestratedStepper()

    def run(
        self,
        run: Run,
        ctx: UpcycleContext,
        *,
        collector: Collector,
        checkpointer: CheckPointer,
        save_every: int,
        resume_state: dict[str, object] | None,
    ) -> None:
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")

        match run.train.phase:
            case TrainPhase.BLOCKWISE:
                self.blockwise.run(
                    run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                )
            case TrainPhase.GLOBAL:
                if getattr(run.train, "orchestrator_enabled", False):
                    self.orchestrated.run(
                        run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                    )
                else:
                    self.global_stepper.run(
                        run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                    )
            case _:
                raise ValueError(
                    f"Unsupported train phase: {run.train.phase}. "
                    f"Expected one of: {[p.value for p in TrainPhase]}"
                )


