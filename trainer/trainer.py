"""Main training loop orchestrator.

The Trainer class reads a manifest file and executes training runs in order.
Each run can be standard training, upcycling, or evaluation. The trainer
handles session management so runs within a group share state.
"""
from __future__ import annotations

from config.manifest import Manifest
from config.mode import Mode
from config.train import TrainPhase
from console import logger
from trainer.upcycle import Upcycle
from trainer.standard import StandardTrainer


class Trainer:
    """Orchestrates training runs from a manifest.

    Iterates through groups and runs, dispatching to the appropriate
    training implementation based on the configured phase.
    """

    def __init__(
        self,
        manifest: Manifest,
    ) -> None:
        """Set up the trainer with a manifest."""
        self.manifest: Manifest = manifest

    def run(self) -> None:
        """Execute all training runs in the manifest.

        Runs are processed in order within each group. The trainer chooses
        between Upcycle (for blockwise/global distillation) and StandardTrainer
        (for regular training) based on the run's phase.
        """
        for group in self.manifest.groups:
            logger.header("Training", f"group={group.name!r}")
            logger.info(f"Scheduled {len(group.runs)} runs")

            # Sessions are managed per-group.
            # For upcycling, we keep the same teacher/student across runs.
            # For standard training, we keep the same model.
            upcycle_session: Upcycle | None = None
            standard_session: StandardTrainer | None = None

            for run in group.runs:
                logger.step(
                    run.id if isinstance(run.id, int) else 0,
                    len(group.runs),
                    f"mode={run.mode} steps={run.steps}",
                )
                if run.mode != Mode.TRAIN:
                    raise ValueError(
                        f"Unsupported mode for run {run.id}: {run.mode}"
                    )
                if run.train is None:
                    raise ValueError(f"Run {run.id} has no train config.")

                if run.train.phase == TrainPhase.STANDARD:
                    if standard_session is None:
                        standard_session = StandardTrainer(
                            manifest=self.manifest,
                            group=group,
                            train=run.train,
                            defaults=self.manifest.defaults,
                        )
                    standard_session.run(run)
                else:
                    # BLOCKWISE or GLOBAL distillation
                    if upcycle_session is None:
                        upcycle_session = Upcycle(
                            manifest=self.manifest,
                            group=group,
                            train=run.train,
                            defaults=self.manifest.defaults,
                        )
                    upcycle_session.run(run)
