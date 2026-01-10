"""Standard (SDPA/GQA) attention layer

This implementation is the reference attention path: it computes dot-product
attention over heads and optionally uses grouped-query attention (GQA) to reduce
KV compute while keeping many query heads.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.layer.attention.base import AttentionBase
from caramba.layer.attention.standard.chunked import StandardSDPAChunked
from caramba.layer.attention.standard.viz import AttentionViz
from caramba.layer.rope import RotaryEmbedding
from caramba.optimizer.attention import AttentionTraining
from caramba.cache.layer import LayerKVCache


class StandardAttentionLayer(AttentionBase):
    """Standard multi-head attention layer

    This layer is “the thing most people mean by attention”: project Q/K/V,
    score keys with QK^T, normalize with softmax, and mix values.
    """
    qkv_proj: nn.Linear
    out_proj: nn.Linear
    rotary: RotaryEmbedding | None
    _scale: float | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        if config.mode == AttentionMode.DECOUPLED:
            raise ValueError("StandardAttentionLayer cannot be constructed with mode=decoupled")

        super().__init__(config)

        self._init_standard(config)
        self._init_common_modules()
        self._viz = AttentionViz()
        self._chunked = StandardSDPAChunked()

    def _init_standard(self, config: AttentionLayerConfig) -> None:
        """Initialize standard projections

        Standard attention learns separate projections for Q/K/V; GQA is a small
        tweak where K/V have fewer heads than Q, which can reduce memory and
        improve speed without changing model dimension.
        """
        d_model = int(config.d_model)
        attn_dim = int(config.attn_dim) if config.attn_dim else d_model
        kv_dim = int(self.n_kv_heads) * int(self.head_dim)

        # Packed QKV projection: one GEMM, then split into q/k/v.
        # This reduces kernel launches and is a common throughput win on GPUs.
        self.qkv_proj = nn.Linear(d_model, attn_dim + 2 * kv_dim, bias=bool(config.bias))
        self.out_proj = nn.Linear(attn_dim, d_model, bias=bool(config.bias))

        if bool(config.rope_enabled):
            self.rotary = RotaryEmbedding(
                int(self.head_dim), base=float(config.rope_base), rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary = None

        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        # Decoupled-only attributes (present on this class as None)
        self.q_sem = None
        self.k_sem = None
        self.q_geo = None
        self.k_geo = None
        self.rotary_sem = None
        self.rotary_geo = None
        self._sem_scale = None
        self._geo_scale = None
        self._v_head_dim = int(self.head_dim)
        self.decoupled_gate_logit = None
        self.decoupled_gate_proj = None
        self.k_sem_null = None
        self.k_geo_null = None
        self.v_null = None

    def _use_attention_training(
        self,
        *,
        qh: Tensor,
        mask: Tensor | None,
        cache: LayerKVCache | None,
        q_chunk: int | None,
        local_window: int | None,
        T: int,
    ) -> bool:
        return bool(
            self.training
            and qh.device.type in ("cuda", "mps")
            # MPS custom kernel limitation: only supports fp16. Fallback to SDPA for float32.
            and not (qh.dtype == torch.float32 and qh.device.type == "mps")
            and mask is None
            and cache is None
            and q_chunk is None
            and local_window is None
            and int(T) > 1
        )

    def _forward_standard(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: LayerKVCache | None,
        pos_offset: int,
        ctx: object | None = None,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
    ) -> tuple[Tensor, LayerKVCache | None]:
        """Compute standard attention output

        The output is a weighted sum of values, where weights come from a
        softmax-normalized similarity between queries and keys.
        """
        _B, T, _ = x.shape
        q_dim = int(self.config.attn_dim) if self.config.attn_dim else int(self.config.d_model)
        kv_dim = int(self.n_kv_heads) * int(self.head_dim)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        qh = self._shape(q, self.head_dim, self.n_heads)
        kh = self._shape(k, self.head_dim, self.n_kv_heads)
        vh = self._shape(v, self.head_dim, self.n_kv_heads)

        if self.rotary is not None:
            qh = self.rotary.rotate(qh, pos_offset)
            kh = self.rotary.rotate(kh, pos_offset)

        qh = self._apply_logit_scale(qh)

        cache_pos = None
        if cache is not None:
            cache_pos = int(cache.pos)
            old_len = int(cache.pos)
            _ = cache.append(self._merge(kh), self._merge(vh))
            if old_len > 0:
                k_all, v_all = cache.get(dtype=qh.dtype)
                kh = self._shape(k_all, self.head_dim, self.n_kv_heads)
                vh = self._shape(v_all, self.head_dim, self.n_kv_heads)

        if self.group_size > 1:
            kh = kh.repeat_interleave(self.group_size, dim=1)
            vh = vh.repeat_interleave(self.group_size, dim=1)

        scale = float(self._scale or 1.0)
        self._viz.record_attention_matrix(
            ctx=ctx,
            layer=self,
            q=qh,
            k=kh,
            scale=scale,
            causal=bool(self.config.is_causal),
        )

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = local_window_override if local_window_override is not None else self.config.local_window
        dropout_p = float(self.config.dropout_p) if self.training else 0.0

        use_chunked = mask is None and (q_chunk is not None or local_window is not None or cache is not None)

        if use_chunked:
            out = self._chunked.run(
                qh=qh,
                kh=kh,
                vh=vh,
                is_causal=bool(self.config.is_causal),
                scale=self._scale,
                pos_offset=int(pos_offset),
                cache_pos=cache_pos,
                q_chunk=int(q_chunk) if q_chunk is not None else int(T),
                local_window=int(local_window) if local_window is not None else None,
                dropout_p=float(dropout_p),
                maybe_summarize_kv=self._maybe_summarize_kv,
            )
        else:
            is_causal = bool(self.config.is_causal) and mask is None and int(T) > 1 and cache is None
            if self._use_attention_training(
                qh=qh,
                mask=mask,
                cache=cache,
                q_chunk=q_chunk,
                local_window=local_window,
                T=int(T),
            ):
                out = AttentionTraining().run(
                    q=qh,
                    k=kh,
                    v=vh,
                    causal=bool(is_causal),
                    scale=scale,
                    attn_mask=None,
                    dropout_p=float(dropout_p),
                )
            else:
                out = F.scaled_dot_product_attention(
                    qh,
                    kh,
                    vh,
                    attn_mask=mask,
                    dropout_p=float(dropout_p),
                    is_causal=bool(is_causal),
                    scale=self._scale,
                )

        y = self.out_proj(self._merge(out))
        self._viz.record_activation_sample(ctx=ctx, layer=self, y=y)
        return y, cache
