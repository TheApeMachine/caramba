"""Initializer component configuration.

Initializers own model construction and weight application for Upcycle.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class InitializerType(str, enum.Enum):
    DEFAULT = "DefaultInitializer"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.initializers"


class DefaultInitializerConfig(Config):
    type: Literal[InitializerType.DEFAULT] = InitializerType.DEFAULT


InitializerConfig: TypeAlias = Annotated[
    DefaultInitializerConfig,
    Field(discriminator="type"),
]

