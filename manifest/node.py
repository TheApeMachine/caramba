"""Node module

The node module contains the node for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, List

from pydantic import (
    BaseModel,
    Field,
    NegativeInt,
    PositiveInt,
    NonNegativeFloat,
    StrictBool,
    Dict,
)

from caramba.manifest.operation import OperationType


class NodeType(str, Enum):
    """Node type enumeration."""
    LINEAR = "linear"
    SPLIT = "split"
    CONCAT = "concat"
    MERGE_HEADS = "merge_heads"
    REPEAT_INTERLEAVE = "repeat_interleave"
    RESHAPE = "reshape"
    SPLIT_SIZES = "split_sizes"
    TRANSPOSE = "transpose"
    VIEW_AS_HEADS = "view_as_heads"


class NodeConfig(BaseModel):
    """A node configuration."""
    d_in: PositiveInt = Field(..., description="Input dimension")
    d_out: PositiveInt = Field(..., description="Output dimension")
    bias: StrictBool = Field(..., description="Bias")
    split_sizes: List[PositiveInt] = Field(..., description="Split sizes")
    dim: NegativeInt = Field(..., description="Dimension")
    num_heads: PositiveInt = Field(..., description="Number of heads")
    base: NonNegativeFloat = Field(..., description="Base")
    variant: str = Field(..., description="Variant")
    repeats: PositiveInt = Field(..., description="Repeats")
    dropout_p: NonNegativeFloat = Field(..., description="Dropout probability")
    is_causal: StrictBool = Field(..., description="Is causal")


class Node(BaseModel):
    """A node configuration."""
    id: NodeType = Field(..., description="Node type")
    op: OperationType = Field(..., description="Operation type")
    inputs: List[str] = Field(..., description="Input keys")
    outputs: List[str] = Field(..., description="Output keys")
    config: Dict[str, Any] = Field(..., description="Configuration")