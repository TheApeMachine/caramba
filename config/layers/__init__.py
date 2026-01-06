"""Layer Configuration

Configurations for each layer type supported by caramba.
"""
from __future__ import annotations

from typing import Annotated, TypeAlias
from pydantic import Field

from caramba.config.layers.diffusion.ddpm import DDPMLayerConfig
from caramba.config.layers.diffusion.ddim import DdimLayerConfig


# Union type for any diffusion layer config, with automatic deserialization
DiffusionLayerConfig: TypeAlias = Annotated[
    DDPMLayerConfig
    | DdimLayerConfig,
    Field(discriminator="type"),
]

# Union type for any diffusion layer config, with automatic deserialization
LayerConfig: TypeAlias = Annotated[
    DiffusionLayerConfig,
    Field(discriminator="type"),
]
