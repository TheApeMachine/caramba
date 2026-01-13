"""Layer module

The layer module contains the layer for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import List
from pydantic import (
    BaseModel, Field, PositiveInt, NonNegativeFloat, StrictBool,
)

from caramba.manifest.operation import Operation


class LayerType(str, Enum):
    """Layer type enumeration."""
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    NORMALIZATION = "normalization"


class Layer(BaseModel):
    """A layer configuration."""
    type: LayerType = Field(..., description="Layer type")
    repeat: PositiveInt = Field(..., description="Repeat count")
    d_model: PositiveInt = Field(..., description="Model dimension")
    eps: NonNegativeFloat = Field(..., description="Epsilon")
    d_in: PositiveInt = Field(..., description="Input dimension")
    d_out: PositiveInt = Field(..., description="Output dimension")
    bias: StrictBool = Field(..., description="Bias")
    layers: List[Layer] = Field(..., description="List of layers")
    operations: List[Operation] = Field(..., description="List of operations")