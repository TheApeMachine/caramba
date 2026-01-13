"""Composable neural network operations

Building blocks for constructing complex neural network architectures through
modular, reusable operations that can be combined in various topologies.
"""
from __future__ import annotations

from caramba.operation.activation import (
    ActivationOperation,
    GELUOperation,
    LeakyReLUOperation,
    ReLUOperation,
    SigmoidOperation,
    SwiGLUOperation,
    TanhOperation,
)
from caramba.operation.attention import AttentionOperation, SDPAOperation, ScaledDotProductAttentionOperation
from caramba.operation.base import Operation
from caramba.operation.cache import (
    CacheOperation,
    InferCtxAttnMaskOperation,
    InferCtxNextCacheOperation,
    InferCtxPosOffsetOperation,
    KVCachePosOperation,
    KVCacheReadOperation,
    KVCacheWriteOperation,
)
from caramba.operation.masking import (
    ApplyMaskOperation,
    CausalMaskLikeOperation,
    CausalMaskOperation,
    CombineMasksOperation,
    MaskingOperation,
)
from caramba.operation.math import (
    AddOperation,
    ClampOperation,
    DropoutOperation,
    InvSqrtDimScaleOperation,
    MathOperation,
    MatMulOperation,
    MulOperation,
    ScaleOperation,
    SoftmaxOperation,
)
from caramba.operation.positional import ApplyRoPEOperation, PositionalOperation
from caramba.operation.shape import (
    ConcatOperation,
    MergeHeadsOperation,
    RepeatInterleaveOperation,
    ReshapeOperation,
    ShapeOperation,
    SplitOperation,
    SplitSizesOperation,
    TransposeOperation,
    ViewAsHeadsOperation,
)
from caramba.operation.tensor import ParameterOperation

__all__ = [
    # Base operations
    "Operation",
    "ShapeOperation",
    # Activation operations
    "ActivationOperation",
    "GELUOperation",
    "LeakyReLUOperation",
    "ReLUOperation",
    "SigmoidOperation",
    "SwiGLUOperation",
    "TanhOperation",
    # Math operations
    "MathOperation",
    "AddOperation",
    "ClampOperation",
    "DropoutOperation",
    "InvSqrtDimScaleOperation",
    "MatMulOperation",
    "MulOperation",
    "ScaleOperation",
    "SoftmaxOperation",
    # Masking operations
    "MaskingOperation",
    "ApplyMaskOperation",
    "CausalMaskOperation",
    "CausalMaskLikeOperation",
    "CombineMasksOperation",
    # Positional operations
    "PositionalOperation",
    "ApplyRoPEOperation",
    # Shape operations
    "ConcatOperation",
    "MergeHeadsOperation",
    "RepeatInterleaveOperation",
    "ReshapeOperation",
    "SplitOperation",
    "SplitSizesOperation",
    "TransposeOperation",
    "ViewAsHeadsOperation",
    # Attention operations
    "AttentionOperation",
    "SDPAOperation",
    "ScaledDotProductAttentionOperation",
    # Tensor source operations
    "ParameterOperation",
    # Cache operations
    "CacheOperation",
    "InferCtxAttnMaskOperation",
    "InferCtxNextCacheOperation",
    "InferCtxPosOffsetOperation",
    "KVCachePosOperation",
    "KVCacheReadOperation",
    "KVCacheWriteOperation",
]
