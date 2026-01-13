"""Checkpointer component configuration.

Checkpointers own checkpoint naming, save/load, and validation logic.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config


class CheckPointerType(str, enum.Enum):
    DEFAULT = "StepwiseCheckPointer"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.checkpointer"

    def py_module(self) -> str:
        """Return the Python module name for this checkpointer type."""
        return "stepwise"


class StepwiseCheckPointerConfig(Config):
    type: Literal[CheckPointerType.DEFAULT] = CheckPointerType.DEFAULT


CheckPointerConfig: TypeAlias = Annotated[
    StepwiseCheckPointerConfig,
    Field(discriminator="type"),
]

