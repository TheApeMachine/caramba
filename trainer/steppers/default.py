"""Default stepper: dispatches to phase-specific steppers."""

from __future__ import annotations

from caramba.config.train import TrainPhase
from caramba.trainer.collectors import Collector
from caramba.trainer.checkpointers import CheckPointer
from caramba.trainer.steppers.blockwise import BlockwiseStepper
from caramba.trainer.steppers.global_stepper import GlobalStepper
from caramba.trainer.steppers.global_orchestrated import GlobalOrchestratedStepper
from caramba.trainer.upcycle_context import UpcycleContext
from caramba.config.run import Run
from caramba.config.stepper import DefaultStepperConfig


class DefaultStepper:
    def __init__(self, config: DefaultStepperConfig) -> None:
        self.config = config
        self._blockwise = BlockwiseStepper()
        self._global = GlobalStepper()
        self._orch = GlobalOrchestratedStepper()

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
                self._blockwise.run(
                    run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                )
            case TrainPhase.GLOBAL:
                if getattr(run.train, "orchestrator_enabled", False):
                    self._orch.run(
                        run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                    )
                else:
                    self._global.run(
                        run, ctx, collector=collector, checkpointer=checkpointer, save_every=save_every, resume_state=resume_state
                    )
            case _:
                raise ValueError(f"Unsupported train phase: {run.train.phase}")


