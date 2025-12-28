"""Collector component configuration.

Collectors are responsible for producing training/validation batches (and
eventually specialized collection flows like verifier batches).
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class CollectorType(str, enum.Enum):
    DEFAULT = "DefaultCollector"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.collectors"


class DefaultCollectorConfig(Config):
    type: Literal[CollectorType.DEFAULT] = CollectorType.DEFAULT


CollectorConfig: TypeAlias = Annotated[
    DefaultCollectorConfig,
    Field(discriminator="type"),
]

