"""MLX Transformer components for DBA experiments.

Provides the building blocks for a Llama-style transformer using MLX,
with support for decoupled bottleneck attention.

Uses MLX built-in optimized operations:
- nn.RMSNorm for layer normalization
- nn.Linear for projections
- nn.Embedding for token embeddings
- nn.silu for SwiGLU activation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from caramba.layer.mlx.attention import DecoupledAttentionMLX, DBAConfig


@dataclass
class TransformerConfig:
    """Configuration for MLX Transformer."""

    d_model: int
    n_layers: int
    n_heads: int

    d_ff: int
    vocab_size: int

    # GQA (defaults to n_heads)
    n_kv_heads: int | None = None
    head_dim: int | None = None  # teacher/baseline head dim (64 for Llama-3.2-1B)

    # DBA value/output per-head dim (paper: sem_head_dim + geo_head_dim = 40)
    v_head_dim: int | None = None

    # DBA semantic compression (per-head)
    sem_head_dim: int = 8

    # DBA geometric compression (per-head) - matches A100 config
    # Default 32 dims/head = geo_dim 1024 / 32 heads
    geo_head_dim: int | None = None  # defaults to 32

    # RoPE settings
    rope_base: float = 500000.0
    rope_scaling: dict[str, Any] | None = None

    # Other settings
    rms_norm_eps: float = 1e-5
    bias: bool = False
    is_causal: bool = True
    tie_embeddings: bool = False
    decoupled_gate: bool = True


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w_gate(x)
        up = self.w_up(x)
        # SwiGLU: swish(gate) * up
        return self.w_down(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Single transformer block with DBA attention and SwiGLU FFN."""

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention norm (using MLX built-in RMSNorm)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # DBA Attention
        dba_config = DBAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            sem_head_dim=config.sem_head_dim,
            v_head_dim=config.v_head_dim,
            geo_head_dim=config.geo_head_dim,
            rope_base=config.rope_base,
            rope_scaling=config.rope_scaling,
            is_causal=config.is_causal,
            bias=config.bias,
            decoupled_gate=config.decoupled_gate,
        )
        self.attention = DecoupledAttentionMLX(dba_config)

        # Pre-FFN norm (using MLX built-in RMSNorm)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # FFN
        self.ffn = SwiGLU(config.d_model, config.d_ff, bias=config.bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
        return_weights: bool = False,
    ) -> tuple[
        mx.array, tuple[mx.array, mx.array, mx.array, mx.array] | None, mx.array | None
    ]:
        """Forward pass.

        Args:
            x: Input tensor (B, T, d_model)
            mask: Optional attention mask
            cache: Optional per-layer attention cache
            return_weights: If True, return attention weights for distillation

        Returns:
            Output tensor (B, T, d_model), updated cache, and optionally attention weights
        """
        # Attention with residual
        h = self.norm1(x)
        attn_out, new_cache, attn_weights = self.attention(
            h, mask=mask, cache=cache, return_weights=return_weights
        )
        x = x + attn_out

        # FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x, new_cache, attn_weights


class DBATransformer(nn.Module):
    """Full transformer model with DBA attention.

    This is a Llama-style decoder-only transformer using decoupled
    bottleneck attention instead of standard multi-head attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = [TransformerBlock(config, i) for i in range(config.n_layers)]

        # Final norm (using MLX built-in RMSNorm)
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Output projection (optionally tied to embeddings)
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[tuple[mx.array, mx.array, mx.array, mx.array]] | None = None,
        return_attention: bool = False,
    ) -> tuple[
        mx.array,
        list[tuple[mx.array, mx.array, mx.array, mx.array]] | None,
        list[mx.array] | None,
    ]:
        """Forward pass.

        Args:
            input_ids: Token IDs (B, T)
            cache: Optional list of per-layer attention caches
            return_attention: If True, return attention weights from all layers

        Returns:
            Logits (B, T, vocab_size), updated cache, and optionally attention weights
        """
        x = self.embed_tokens(input_ids)

        use_cache = cache is not None
        cache_in = cache
        if cache_in == []:
            cache_in = [None] * len(self.layers)

        new_cache = []
        all_attn_weights = [] if return_attention else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache_in[i] if cache_in is not None else None
            x, updated_cache, attn_weights = layer(
                x, cache=layer_cache, return_weights=return_attention
            )
            if use_cache:
                new_cache.append(updated_cache)
            if return_attention:
                all_attn_weights.append(attn_weights)

        x = self.norm(x)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings: use transpose of embedding matrix
            logits = x @ self.embed_tokens.weight.T

        return logits, new_cache if use_cache else None, all_attn_weights


def create_llama_dba_config(
    d_model: int = 2048,
    n_layers: int = 16,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    d_ff: int = 8192,
    vocab_size: int = 128256,
    sem_head_dim: int = 8,
    v_head_dim: int | None = None,
    geo_head_dim: int = 32,  # Compressed to match A100 DBA (1024 / 32 heads)
    rope_base: float = 500000.0,
    **kwargs: Any,
) -> TransformerConfig:
    """Create a Llama-3.2-1B compatible DBA config.

    This config uses compressed geometric dimensions matching A100 DBA:
    - sem_head_dim: 8 (semantic path - content routing)
    - geo_head_dim: 32 (geometric path - position routing)
    - head_dim: 64 (V/O projections - full Llama dimensions)
    """
    return TransformerConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        vocab_size=vocab_size,
        sem_head_dim=sem_head_dim,
        v_head_dim=v_head_dim,
        geo_head_dim=geo_head_dim,
        rope_base=rope_base,
        **kwargs,
    )
