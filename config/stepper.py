"""Stepper component configuration.

Steppers own the training loops (blockwise/global/orchestrated) and are reusable
in other sessions as nested orchestrators.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class StepperType(str, enum.Enum):
    DEFAULT = "DefaultStepper"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.steppers"


class DefaultStepperConfig(Config):
    type: Literal[StepperType.DEFAULT] = StepperType.DEFAULT


StepperConfig: TypeAlias = Annotated[
    DefaultStepperConfig,
    Field(discriminator="type"),
]

