"""Tensor shape manipulation operations

Essential operations for transforming tensor dimensions in neural networks,
used throughout attention mechanisms, convolutional layers, and model architectures.
"""
from __future__ import annotations

from caramba.operation.shape.base import ShapeOperation
from caramba.operation.shape.concat import ConcatOperation
from caramba.operation.shape.merge_heads import MergeHeadsOperation
from caramba.operation.shape.repeat_interleave import RepeatInterleaveOperation
from caramba.operation.shape.reshape import ReshapeOperation
from caramba.operation.shape.split import SplitOperation
from caramba.operation.shape.split_sizes import SplitSizesOperation
from caramba.operation.shape.transpose import TransposeOperation
from caramba.operation.shape.view_as_heads import ViewAsHeadsOperation

__all__ = [
    "ShapeOperation",
    "ConcatOperation",
    "MergeHeadsOperation",
    "RepeatInterleaveOperation",
    "ReshapeOperation",
    "SplitOperation",
    "SplitSizesOperation",
    "TransposeOperation",
    "ViewAsHeadsOperation",
]
