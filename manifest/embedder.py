"""Embedder module

The embedder module contains the embedder for the manifest.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, PositiveInt


class EmbedderType(str, Enum):
    """Embedder type enumeration."""
    NONE = "none"
    TOKEN = "token"
    PATCH = "patch"


class Embedder(BaseModel):
    """An embedder configuration."""
    type: EmbedderType = Field(..., description="Embedder type")
    vocab_size: PositiveInt = Field(..., description="Vocabulary size")
    d_model: PositiveInt = Field(..., description="Embedding dimension")


class NoEmbedder(Embedder):
    """A no embedder configuration."""
    type: EmbedderType = EmbedderType.NONE