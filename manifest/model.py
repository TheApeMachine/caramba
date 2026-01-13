"""Model manifest

Holds values on a model for the manifest
"""
from __future__ import annotations

from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat


class Model(BaseModel):
    """A model configuration."""
    d_model: PositiveInt = Field(..., description="Model dimension")
    n_layers: PositiveInt = Field(..., description="Number of layers")
    n_heads: PositiveInt = Field(..., description="Number of attention heads")
    n_kv_heads_gqa: PositiveInt = Field(..., description="Number of KV heads for GQA")
    d_ff: PositiveInt = Field(..., description="Feed-forward dimension")
    vocab_size: PositiveInt = Field(..., description="Vocabulary size")
    rope_base: NonNegativeFloat = Field(..., description="RoPE base frequency")