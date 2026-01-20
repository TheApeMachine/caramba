"""Diffusion Layer Configurations

Configurations for the DDPM sampler.
"""
from __future__ import annotations

from typing import Literal

from config.layers.diffusion import (
    DiffusionLayerConfig,
    DiffusionLayerType,
)


class DDPMLayerConfig(DiffusionLayerConfig):
    """Configuration for the DDPM sampler.

    The configuration defines all the parameters for the DDPM sampler.
    """
    type: Literal[DiffusionLayerType.DDPM] = DiffusionLayerType.DDPM