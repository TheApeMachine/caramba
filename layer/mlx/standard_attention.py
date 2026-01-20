"""MLX Standard Llama Attention - Teacher model for distillation.

This implements standard multi-head attention matching Llama's architecture,
used as a teacher to distill attention patterns into DBA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class StandardAttentionConfig:
    """Configuration for standard Llama attention with GQA support."""

    d_model: int
    n_heads: int  # Number of query heads
    n_kv_heads: int | None = None  # Number of key/value heads (GQA), defaults to n_heads
    head_dim: int | None = None  # Defaults to d_model // n_heads
    rope_base: float = 10000.0
    is_causal: bool = True
    bias: bool = False
    rope_scaling: dict[str, Any] | None = None

    @property
    def computed_head_dim(self) -> int:
        return self.head_dim if self.head_dim else self.d_model // self.n_heads

    @property
    def computed_n_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads else self.n_heads


class StandardAttentionMLX(nn.Module):
    """Standard Llama-style multi-head attention with GQA support.

    Used as teacher model to provide target attention weights for DBA distillation.
    Can return attention weights for KL/MSE loss against DBA.

    Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.
    """

    def __init__(self, config: StandardAttentionConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.computed_n_kv_heads
        self.d_model = config.d_model
        self.head_dim = config.computed_head_dim

        # GQA: Q has n_heads, K/V have n_kv_heads
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # Number of query heads per KV head (for GQA expansion)
        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard Q/K/V projections (K/V may be smaller for GQA)
        self.q_proj = nn.Linear(config.d_model, q_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(q_dim, config.d_model, bias=config.bias)

        # RoPE config
        self._rope_dims = self.head_dim
        self._rope_base = config.rope_base
        self._rope_traditional = False

        # Attention scale
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def _shape_q(self, x: mx.array) -> mx.array:
        """Reshape Q: (B, T, n_heads*D) -> (B, n_heads, T, D)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _shape_kv(self, x: mx.array) -> mx.array:
        """Reshape K/V: (B, T, n_kv_heads*D) -> (B, n_kv_heads, T, D)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge(self, x: mx.array) -> mx.array:
        """Reshape (B, H, T, D) -> (B, T, H*D)."""
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        return_weights: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None, mx.array | None]:
        """Forward pass with optional attention weight return.

        Args:
            x: Input tensor (B, T, d_model)
            mask: Optional attention mask
            cache: Optional (k, v) cache for inference
            return_weights: If True, return attention weights for distillation

        Returns:
            - Output tensor (B, T, d_model)
            - Updated cache
            - Attention weights (B, H, T, T) if return_weights=True, else None
        """
        B, T, _ = x.shape

        # Compute Q/K/V with different head counts
        q = self._shape_q(self.q_proj(x))  # (B, n_heads, T, D)
        k = self._shape_kv(self.k_proj(x))  # (B, n_kv_heads, T, D)
        v = self._shape_kv(self.v_proj(x))  # (B, n_kv_heads, T, D)

        # Position offset for caching
        pos_offset = 0 if cache is None else cache[0].shape[2]

        # Apply RoPE
        q = mx.fast.rope(
            q, dims=self._rope_dims, traditional=self._rope_traditional,
            base=self._rope_base, scale=1.0, offset=pos_offset,
        )
        k = mx.fast.rope(
            k, dims=self._rope_dims, traditional=self._rope_traditional,
            base=self._rope_base, scale=1.0, offset=pos_offset,
        )

        # Handle cache
        new_cache = None
        if cache is not None:
            k_cached, v_cached = cache
            k = mx.concatenate([k_cached, k], axis=2)
            v = mx.concatenate([v_cached, v], axis=2)
        new_cache = (k, v)

        # GQA: Expand K/V heads to match Q heads by repeating
        # k shape: (B, n_kv_heads, S, D) -> (B, n_heads, S, D)
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self._scale  # (B, n_heads, T, S)

        # Apply causal mask
        if self.config.is_causal:
            S = k.shape[2]
            # Create causal mask: positions can only attend to earlier positions
            causal_mask = mx.triu(mx.full((T, S), -1e9), k=S - T + 1)
            scores = scores + causal_mask

        # Softmax to get attention weights
        attn_weights = mx.softmax(scores, axis=-1)  # (B, n_heads, T, S)

        # Apply attention to values
        out = attn_weights @ v  # (B, n_heads, T, D)

        # Project output
        y = self.out_proj(self._merge(out))

        if return_weights:
            return y, new_cache, attn_weights
        return y, new_cache, None


class TeacherTransformerBlock(nn.Module):
    """Transformer block using standard attention (for teacher model)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int | None = None,
        rope_base: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()

        # Pre-attention norm
        self.norm1 = nn.RMSNorm(d_model, eps=rms_norm_eps)

        # Standard Attention with GQA support
        attn_config = StandardAttentionConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            bias=bias,
        )
        self.attention = StandardAttentionMLX(attn_config)

        # Pre-FFN norm
        self.norm2 = nn.RMSNorm(d_model, eps=rms_norm_eps)

        # FFN (SwiGLU)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
        return_weights: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None, mx.array | None]:
        """Forward pass.

        Returns:
            - Output tensor (B, T, d_model)
            - Updated cache
            - Attention weights if return_weights=True
        """
        # Attention with residual
        h = self.norm1(x)
        attn_out, new_cache, attn_weights = self.attention(
            h, cache=cache, return_weights=return_weights
        )
        x = x + attn_out

        # FFN with residual
        h = self.norm2(x)
        gate = self.w_gate(h)
        up = self.w_up(h)
        x = x + self.w_down(nn.silu(gate) * up)

        return x, new_cache, attn_weights


class TeacherModel(nn.Module):
    """Full teacher transformer for attention distillation.

    This model uses standard attention and can return per-layer attention
    weights for distilling into a DBA student model.

    Supports GQA (Grouped Query Attention) used in Llama 3.2.
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_layers: int = 16,
        n_heads: int = 32,
        n_kv_heads: int = 8,  # Llama 3.2 1B uses GQA with 8 KV heads
        d_ff: int = 8192,
        vocab_size: int = 128256,
        rope_base: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = [
            TeacherTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                rope_base=rope_base,
                rms_norm_eps=rms_norm_eps,
                bias=bias,
            )
            for _ in range(n_layers)
        ]

        # Final norm
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
        return_attention: bool = False,
    ) -> tuple[mx.array, list | None, list[mx.array] | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs (B, T)
            cache: Optional list of (k, v) caches per layer
            return_attention: If True, return attention weights from all layers

        Returns:
            - Logits (B, T, vocab_size)
            - Updated cache
            - List of attention weights per layer if return_attention=True
        """
        x = self.embed_tokens(input_ids)

        new_cache = []
        all_attn_weights: list[mx.array] | None = [] if return_attention else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache, attn_weights = layer(
                x, cache=layer_cache, return_weights=return_attention
            )
            if cache is not None:
                new_cache.append(updated_cache)
            if return_attention:
                assert all_attn_weights is not None  # Type narrowing
                assert attn_weights is not None  # Should be non-None when return_weights=True
                all_attn_weights.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_cache if cache is not None else None, all_attn_weights
