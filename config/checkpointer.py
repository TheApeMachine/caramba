"""Checkpointer component configuration.

Checkpointers own checkpoint naming, save/load, and validation logic.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config


class CheckPointerType(str, enum.Enum):
    DEFAULT = "DefaultCheckPointer"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.checkpointers"


class DefaultCheckPointerConfig(Config):
    type: Literal[CheckPointerType.DEFAULT] = CheckPointerType.DEFAULT


CheckPointerConfig: TypeAlias = Annotated[
    DefaultCheckPointerConfig,
    Field(discriminator="type"),
]

