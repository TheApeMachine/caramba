"""Embedder configuration: how tokens become vectors.

Models can use different embedding strategies:
- NONE: Input is already embedded (for pre-processed data)
- TOKEN: Standard learned embedding table (token ID → vector)
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from config import PositiveInt


class EmbedderType(str, enum.Enum):
    """Type of embedding layer."""

    NONE = "none"
    TOKEN = "token"
    PATCH = "patch"


class _EmbedderConfigBase(BaseModel):
    """Base class for embedder configs."""

    pass


class NoEmbedderConfig(_EmbedderConfigBase):
    """Config for no embedding (input is already vectors)."""

    type: Literal[EmbedderType.NONE] = EmbedderType.NONE


class TokenEmbedderConfig(_EmbedderConfigBase):
    """Config for standard token embedding table.

    Maps token IDs to d_model-dimensional vectors via a learned
    embedding matrix of shape (vocab_size, d_model).
    """

    type: Literal[EmbedderType.TOKEN] = EmbedderType.TOKEN
    vocab_size: PositiveInt
    d_model: PositiveInt


class PatchEmbedderConfig(_EmbedderConfigBase):
    """Config for image patch embedding (ViT-style).

    Divides an image into non-overlapping patches and projects them
    to d_model-dimensional vectors via a 2D convolution.
    """

    type: Literal[EmbedderType.PATCH] = EmbedderType.PATCH
    img_size: PositiveInt = 224
    patch_size: PositiveInt = 16
    in_channels: PositiveInt = 3
    d_model: PositiveInt


# Union type for any embedder config
EmbedderConfig: TypeAlias = Annotated[
    NoEmbedderConfig | TokenEmbedderConfig | PatchEmbedderConfig,
    Field(discriminator="type"),
]
