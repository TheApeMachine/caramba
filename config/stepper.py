"""Stepper component configuration.

Steppers own the training loops (blockwise/global/orchestrated) and are reusable
in other sessions as nested orchestrators.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config


class StepperType(str, enum.Enum):
    """Types of steppers available for training coordination"""
    PHASE_DISPATCHER = "PhaseDispatcherStepper"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.steppers"


class PhaseDispatcherConfig(Config):
    """Configuration for phase dispatcher stepper

    The phase dispatcher routes training to appropriate phase-specific
    steppers based on the training phase (blockwise, global, orchestrated).
    """
    type: Literal[StepperType.PHASE_DISPATCHER] = StepperType.PHASE_DISPATCHER


StepperConfig: TypeAlias = Annotated[
    PhaseDispatcherConfig,
    Field(discriminator="type"),
]

