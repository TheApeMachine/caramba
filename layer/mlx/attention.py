"""MLX Decoupled Bottleneck Attention (DBA) layer.

This is a Llama-compatible DBA implementation that supports GQA (Grouped Query
Attention) so we can do behavior-preserving “surgery” on a pretrained model.

Key design goals:
- Keep the *geometric* path shape-compatible with Llama attention (including GQA)
- Apply RoPE only on the geometric path
- Keep the *semantic* path aggressively compressed (small per-head dim)
- Avoid weight slicing/padding hacks when loading pretrained weights

Note: For correctness and alignment with pretrained behavior, this implementation
uses explicit matmul+softmax rather than fused SDPA. Performance work can follow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DBAConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int | None = None  # defaults to n_heads

    # Llama head dim (used for compatibility with teacher weights / init modes)
    # Defaults to d_model // n_heads (64 for Llama-3.2-1B)
    head_dim: int | None = None

    # Semantic compression (per-head) - small for content-based routing
    sem_head_dim: int = 8

    # Value/O per-head dim.
    # Paper spec uses v_head_dim = sem_head_dim + geo_head_dim (40 for 8+32).
    # For behavior-preserving copy-V/O init, set v_head_dim = head_dim.
    v_head_dim: int | None = None

    # Geometric compression (per-head) - for position-based routing
    # Default 32 matches A100 DBA config (geo_dim=1024 / 32 heads)
    # Set to head_dim (64) to match full Llama Q/K for copy_vo mode
    geo_head_dim: int | None = None  # defaults to 32

    # RoPE settings (geometric path only)
    rope_base: float = 500000.0
    rope_scaling: dict[str, Any] | None = None
    is_causal: bool = True
    bias: bool = False

    # Optional learnable gate mixing semantic vs geometric scores
    decoupled_gate: bool = True

    @property
    def computed_n_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def computed_head_dim(self) -> int:
        """Teacher/baseline head dim (64 for Llama-3.2-1B)."""
        return (
            self.head_dim
            if self.head_dim is not None
            else (self.d_model // self.n_heads)
        )

    @property
    def computed_v_head_dim(self) -> int:
        """Per-head value/output dim for DBA.

        Defaults to sem_head_dim + geo_head_dim to match the paper's d_attn.
        """
        if self.v_head_dim is not None:
            return int(self.v_head_dim)
        return int(self.sem_head_dim + self.computed_geo_head_dim)

    @property
    def computed_geo_head_dim(self) -> int:
        """Head dim for geometric Q/K (compressed by default)."""
        return self.geo_head_dim if self.geo_head_dim is not None else 32

    @property
    def q_dim(self) -> int:
        """Teacher/baseline Q dimension (full Llama)."""
        return self.n_heads * self.computed_head_dim

    @property
    def v_q_dim(self) -> int:
        """DBA output projection input dim (n_heads * v_head_dim)."""
        return self.n_heads * self.computed_v_head_dim

    @property
    def geo_q_dim(self) -> int:
        """Geometric Q dimension (compressed)."""
        return self.n_heads * self.computed_geo_head_dim

    @property
    def geo_kv_dim(self) -> int:
        """Geometric K dimension (compressed, GQA)."""
        return self.computed_n_kv_heads * self.computed_geo_head_dim

    @property
    def kv_dim(self) -> int:
        """Teacher/baseline V dimension (full Llama, GQA)."""
        return self.computed_n_kv_heads * self.computed_head_dim

    @property
    def v_kv_dim(self) -> int:
        """DBA value projection output dim (n_kv_heads * v_head_dim)."""
        return self.computed_n_kv_heads * self.computed_v_head_dim

    @property
    def sem_q_dim(self) -> int:
        return self.n_heads * self.sem_head_dim

    @property
    def sem_kv_dim(self) -> int:
        return self.computed_n_kv_heads * self.sem_head_dim


class Llama3RotaryEmbedding(nn.Module):
    """Llama 3 compatible RoPE with piecewise scaling (MLX version).

    IMPORTANT: inv_freq is computed for a *full* head_dim (e.g., teacher head_dim=64).
    At apply time, we may pass tensors with smaller D (e.g., geo_head_dim=32).
    We slice inv_freq to D//2 so the frequency schedule matches the teacher's
    first dimensions.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 500000.0,
        scaling: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.full_head_dim = head_dim
        self.base = base

        # Calculate frequencies for full head_dim
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )

        # Apply Llama 3 scaling if requested
        if scaling and scaling.get("rope_type") == "llama3":
            factor = float(scaling.get("factor", 8.0))
            low_freq_factor = float(scaling.get("low_freq_factor", 1.0))
            high_freq_factor = float(scaling.get("high_freq_factor", 4.0))
            old_context_len = float(
                scaling.get("original_max_position_embeddings", 8192)
            )

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = inv_freq / factor

            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smooth = mx.maximum(0.0, mx.minimum(1.0, smooth))

            is_low = wavelen > low_freq_wavelen
            is_high = wavelen < high_freq_wavelen

            # MLX where equivalent
            inv_freq = mx.where(is_low, inv_freq_llama, inv_freq)
            inv_freq = mx.where(
                is_high, inv_freq, (1 - smooth) * inv_freq_llama + smooth * inv_freq
            )

        self.inv_freq = inv_freq  # shape: (full_head_dim/2,)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (B, H, T, D) where D may be <= full_head_dim
        B, H, T, D = x.shape
        if D % 2 != 0:
            raise ValueError("RoPE requires even last-dim")

        # Slice teacher-defined frequencies to match D
        inv = self.inv_freq[: (D // 2)]  # (D/2,)

        positions = mx.arange(offset, offset + T, dtype=mx.float32)  # (T,)
        freqs = positions[:, None] * inv[None, :]  # (T, D/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (T, D)

        cos = mx.cos(emb)[None, None, :, :]  # (1,1,T,D)
        sin = mx.sin(emb)[None, None, :, :]  # (1,1,T,D)

        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class DecoupledAttentionMLX(nn.Module):
    """DBA attention with Llama-compatible shapes (supports GQA).

    Scores are additive:
      score = score_sem / sqrt(d_sem) + score_geo / sqrt(d_geo)

    RoPE is applied only to the geometric path.

    Cache stores KV in *kv-head* form: (B, n_kv_heads, S, dim).
    """

    def __init__(self, config: DBAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.computed_n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        self.d_model = config.d_model

        # Teacher/baseline head dim (used by some init modes)
        self.teacher_head_dim = config.computed_head_dim

        # DBA value head dim (paper: 8+32=40)
        self.v_head_dim = config.computed_v_head_dim

        # Q/K head dims
        self.geo_head_dim = config.computed_geo_head_dim
        self.sem_head_dim = int(config.sem_head_dim)

        if self.geo_head_dim % 2 != 0:
            raise ValueError("RoPE requires even geo_head_dim")
        if self.sem_head_dim <= 0:
            raise ValueError("sem_head_dim must be > 0")

        # Semantic projections (compressed)
        self.q_sem = nn.Linear(self.d_model, config.sem_q_dim, bias=config.bias)
        self.k_sem = nn.Linear(self.d_model, config.sem_kv_dim, bias=config.bias)

        # Geometric projections (compressed - matches A100 DBA)
        self.q_geo = nn.Linear(self.d_model, config.geo_q_dim, bias=config.bias)
        self.k_geo = nn.Linear(self.d_model, config.geo_kv_dim, bias=config.bias)

        # Value/output projections (DBA dims)
        self.v_proj = nn.Linear(self.d_model, config.v_kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.v_q_dim, self.d_model, bias=config.bias)

        # RoPE for geometric path (uses geo_head_dim)
        self._rope_dims = self.geo_head_dim
        self._rope_base = float(config.rope_base)
        self._rope_traditional = False

        # Use custom RoPE if scaling is provided (Llama 3 support)
        # We assume if rope_scaling is present, it might be Llama 3 type
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = None

        # Use teacher head dim for RoPE frequencies when geo is compressed.
        # This keeps the frequency schedule consistent with the pretrained model.
        rope_freq_dim = self.teacher_head_dim if self.geo_head_dim != self.teacher_head_dim else self.geo_head_dim

        # Always use the custom RoPE when frequencies must match teacher dims OR llama3 scaling is present.
        if self.rope_scaling or (self.geo_head_dim != self.teacher_head_dim):
            self.rotary_emb = Llama3RotaryEmbedding(
                rope_freq_dim, base=self._rope_base, scaling=self.rope_scaling
            )

        # Optional gate on semantic score contribution (g=0 => geometric-only)
        if config.decoupled_gate:
            # Neutral start; semantic weights are small-initialized in surgery.
            self.gate_logit = mx.full((self.n_heads,), -4.0)  # sigmoid ~ 0.018
        else:
            self.gate_logit = None

    def _shape_q(self, x: mx.array, head_dim: int) -> mx.array:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

    def _shape_kv(self, x: mx.array, head_dim: int) -> mx.array:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    def _repeat_kv(self, x: mx.array) -> mx.array:
        # (B, n_kv_heads, S, D) -> (B, n_heads, S, D)
        if self.n_rep == 1:
            return x
        return mx.repeat(x, self.n_rep, axis=1)

    def _merge(self, x: mx.array) -> mx.array:
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

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

        Cache is (k_sem, k_geo, v, dummy) to keep tuple stability; dummy is reserved.
        Shapes (before repetition):
          k_sem: (B, n_kv_heads, S, sem_head_dim)
          k_geo: (B, n_kv_heads, S, geo_head_dim)
          v:     (B, n_kv_heads, S, v_head_dim)
        """
        B, T, _ = x.shape

        # Project
        q_sem = self._shape_q(self.q_sem(x), self.sem_head_dim)  # (B, H, T, d_sem)
        k_sem = self._shape_kv(self.k_sem(x), self.sem_head_dim)  # (B, H_kv, T, d_sem)

        q_geo = self._shape_q(self.q_geo(x), self.geo_head_dim)  # (B, H, T, d_geo)
        k_geo = self._shape_kv(self.k_geo(x), self.geo_head_dim)  # (B, H_kv, T, d_geo)

        v = self._shape_kv(self.v_proj(x), self.v_head_dim)  # (B, H_kv, T, d_v)

        # Cache concat + RoPE offset
        pos_offset = 0
        if cache is not None:
            k_sem_cached, k_geo_cached, v_cached, _ = cache
            pos_offset = int(k_geo_cached.shape[2])
            k_sem = mx.concatenate([k_sem_cached, k_sem], axis=2)
            k_geo = mx.concatenate([k_geo_cached, k_geo], axis=2)
            v = mx.concatenate([v_cached, v], axis=2)

        # Apply RoPE ONLY to geometric
        if self.rotary_emb is not None:
            q_geo = self.rotary_emb(q_geo, offset=pos_offset)
            k_geo = self.rotary_emb(k_geo, offset=pos_offset)
        else:
            q_geo = mx.fast.rope(
                q_geo,
                dims=self._rope_dims,
                traditional=self._rope_traditional,
                base=self._rope_base,
                scale=1.0,
                offset=pos_offset,
            )
            k_geo = mx.fast.rope(
                k_geo,
                dims=self._rope_dims,
                traditional=self._rope_traditional,
                base=self._rope_base,
                scale=1.0,
                offset=pos_offset,
            )

        # Expand KV heads to Q heads
        k_sem_rep = self._repeat_kv(k_sem)
        k_geo_rep = self._repeat_kv(k_geo)
        v_rep = self._repeat_kv(v)

        # Scores
        s_sem = (q_sem @ k_sem_rep.transpose(0, 1, 3, 2)) * (
            1.0 / math.sqrt(self.sem_head_dim)
        )
        s_geo = (q_geo @ k_geo_rep.transpose(0, 1, 3, 2)) * (
            1.0 / math.sqrt(self.geo_head_dim)
        )

        if self.gate_logit is not None:
            g = mx.sigmoid(self.gate_logit).reshape(1, -1, 1, 1)
            scores = s_geo + g * s_sem
        else:
            scores = s_sem + s_geo

        # Masking
        if self.config.is_causal:
            S = k_geo.shape[2]
            causal_mask = mx.triu(mx.full((T, S), -1e9), k=S - T + 1)
            scores = scores + causal_mask
        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v_rep  # (B, H, T, head_dim)

        y = self.out_proj(self._merge(out))

        new_cache = (k_sem, k_geo, v, mx.array(0.0))
        if return_weights:
            return y, new_cache, attn
        return y, new_cache, None
