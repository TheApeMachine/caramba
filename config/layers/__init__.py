"""Layer Configuration

Configurations for each layer type supported by caramba.
"""
from __future__ import annotations

from typing import Annotated, TypeAlias
from pydantic import Field

from config.layers.diffusion.ddpm import DDPMLayerConfig
from config.layers.diffusion.ddim import DdimLayerConfig


# Configs specific to diffusion layers; supports automatic deserialization into
# concrete diffusion layer types via the layer factory/discriminated union.
DiffusionLayerConfig: TypeAlias = Annotated[
    DDPMLayerConfig
    | DdimLayerConfig,
    Field(discriminator="type"),
]

# Broader union type for any layer config (including non-diffusion layers) used
# by the layer factory/deserializer. Today this is identical to DiffusionLayerConfig.
LayerConfig: TypeAlias = DiffusionLayerConfig
