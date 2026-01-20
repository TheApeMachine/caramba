"""Weight initialization configuration."""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class WeightInitType(str, enum.Enum):
    """Type of weight initialization strategy."""

    GPT2 = "GPT2Initializer"
    NONE = "NoInitializer"  # Default PyTorch init

    @classmethod
    def from_str(cls, s: str) -> "WeightInitType":
        return cls(s)

    @staticmethod
    def module_name() -> str:
        return "caramba.initializers"


class GPT2InitConfig(Config):
    """Configuration for GPT-2 style initialization."""
    type: Literal[WeightInitType.GPT2] = WeightInitType.GPT2
    n_layers: int = 12


class NoInitConfig(Config):
    """Configuration for default PyTorch initialization (do nothing)."""
    type: Literal[WeightInitType.NONE] = WeightInitType.NONE


WeightInitConfig: TypeAlias = Annotated[
    GPT2InitConfig | NoInitConfig,
    Field(discriminator="type"),
]
