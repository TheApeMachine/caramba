"""MLX layer implementations.

This module provides MLX-native implementations of caramba layers,
optimized for Apple Silicon via the MLX framework.

Uses MLX built-in optimized operations:
- mx.fast.rope for rotary position embeddings
- nn.RMSNorm for layer normalization
"""

from __future__ import annotations

from caramba.layer.mlx.attention import DecoupledAttentionMLX, DBAConfig
from caramba.layer.mlx.transformer import (
    DBATransformer,
    TransformerConfig,
    TransformerBlock,
    SwiGLU,
    create_llama_dba_config,
)

__all__ = [
    "DecoupledAttentionMLX",
    "DBAConfig",
    "DBATransformer",
    "TransformerConfig",
    "TransformerBlock",
    "SwiGLU",
    "create_llama_dba_config",
]
