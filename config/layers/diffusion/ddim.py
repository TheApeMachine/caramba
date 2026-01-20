"""DDIM sampler configuration

Configuration for the DDIM sampler.
"""
from __future__ import annotations

from typing import Literal

from config.layers.diffusion import (
    DiffusionLayerConfig, DiffusionLayerType
)


class DdimLayerConfig(DiffusionLayerConfig):
    """Configuration for the DDIM sampler.

    The configuration defines all the parameters for the DDIM sampler.
    """
    type: Literal[DiffusionLayerType.DDIM] = DiffusionLayerType.DDIM
