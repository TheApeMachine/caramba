"""Diffusion Layer Configurations

Configurations for the diffusion layers.
"""
from __future__ import annotations

import enum
from typing import Literal

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    Probability,
)

from caramba.config import Config


class DiffusionLayerType(str, enum.Enum):
    """Diffusion Layer Type

    Provides a typed enum for layer types.
    """
    DDPM = "DDPM" # Denoising Diffusion Probabilistic Model
    DDIM = "DDIM" # Denoising Diffusion Implicit Model


class DiffusionLayerConfig(Config):
    """Diffusion Layer Configuration

    The configuration defines all the parameters for the diffusion layer.
    """
    type: DiffusionLayerType = Field(discriminator="type")
    hidden_size: PositiveInt
    time_embed_dim: PositiveInt
    mlp_mult: PositiveInt
    cfg_dropout_p: Probability
    cfg_guidance_scale: PositiveFloat
    scheduler: Literal["ddim", "ddpm", "dpm"]
    loss_weight: PositiveFloat
    timesteps: PositiveInt
    infer_steps: PositiveInt