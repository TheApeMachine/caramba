"""Standard training session types.

These small types are used by the standard (non-upcycling) trainer stack. They
model the immutable "session bundle" that a training loop operates on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from caramba.trainer.collector.training import TrainHook
from caramba.config.defaults import Defaults
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig
from caramba.instrumentation import RunLogger


class TrainStepper(Protocol):
    def run(self, session: "TrainSession") -> None: ...


@dataclass(frozen=True, slots=True)
class TrainSession:
    """Immutable bundle of objects required for a training run."""

    defaults: Defaults
    target: ExperimentTargetConfig
    run: Run
    train: TrainConfig

    dataset_comp: object
    system: object
    objective: object

    checkpoint_dir: Path
    run_logger: RunLogger
    hooks: list[TrainHook] | None = None
