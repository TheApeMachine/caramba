"""Training stepper package

Steppers are small, composable orchestrators that own training loops.
They are intentionally kept as thin as possible, delegating concrete concerns
like loss computation, AMP policy, validation evaluation, and checkpointing to
specialized collaborators.
"""
from __future__ import annotations

from caramba.trainer.stepper.base import Stepper
from caramba.trainer.stepper.builder import StepperBuilder
from caramba.trainer.stepper.session import TrainSession, TrainStepper

__all__ = [
    "Stepper",
    "StepperBuilder",
    "TrainSession",
    "TrainStepper",
]
