"""Operation module

The operation module contains the operation for the manifest.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, PositiveInt, StrictBool

from caramba.manifest.graph import Graph


class OperationType(str, Enum):
    """Operation type enumeration."""
    ATTENTION = "attention"
    LINEAR = "linear"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    POSITIONAL_ENCODING = "positional_encoding"
    SHAPE = "shape"
    CONCAT = "concat"
    MERGE_HEADS = "merge_heads"
    REPEAT_INTERLEAVE = "repeat_interleave"
    RESHAPE = "reshape"
    SPLIT = "split"
    TRANSPOSE = "transpose"
    VIEW_AS_HEADS = "view_as_heads"
    SPLIT_SIZES = "split_sizes"


class Operation(BaseModel):
    """An operation configuration."""
    type: OperationType = Field(..., description="Operation type")
    d_in: PositiveInt = Field(..., description="Input dimension")
    d_out: PositiveInt = Field(..., description="Output dimension")
    bias: StrictBool = Field(..., description="Bias")
    graph: Graph = Field(..., description="Graph")