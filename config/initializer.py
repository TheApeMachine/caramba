"""Initializer component configuration.

Initializers own model construction and weight application for Upcycle.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config


class InitializerType(str, enum.Enum):
    """Types of initializers available for model construction"""
    UPCYCLE = "UpcycleInitializer"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.initializers"


class UpcycleInitializerConfig(Config):
    """Configuration for upcycle initializer

    Loads pretrained teacher checkpoint and initializes student model with
    architecture surgery for knowledge distillation.
    """
    type: Literal[InitializerType.UPCYCLE] = InitializerType.UPCYCLE


InitializerConfig: TypeAlias = Annotated[
    UpcycleInitializerConfig,
    Field(discriminator="type"),
]

