"""Decoupled (DBA) attention layer

DBA splits attention into semantic and geometric channels and recombines them,
which is a way to test whether different “types” of relationships benefit from
different representations and inductive biases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.layer.attention.base import AttentionBase
from caramba.layer.attention.decoupled.chunked import DecoupledSDPAChunked
from caramba.layer.attention.decoupled.decode import DecoupledDecode
from caramba.layer.attention.decoupled.memory import DecoupledMemorySummarizer
from caramba.layer.attention.decoupled.setup import DecoupledSetup
from caramba.layer.attention.decoupled.viz import DecoupledAttentionViz
from caramba.layer.rope import RotaryEmbedding
from caramba.optimizer.dba_attention_triton import DecoupledAttentionTraining

if TYPE_CHECKING:
    from caramba.cache.decoupled import DecoupledLayerKVCache


class DecoupledAttentionLayer(AttentionBase, DecoupledSetup, DecoupledMemorySummarizer):
    """Decoupled bottleneck attention layer

    The goal is to give the model two specialized sub-attentions and a learned
    way to mix them, instead of forcing one attention space to represent
    everything at once.
    """

    q_sem: nn.Linear | None
    k_sem: nn.Linear | None
    q_geo: nn.Linear | None
    k_geo: nn.Linear | None
    v_proj: nn.Linear
    out_proj: nn.Linear
    rotary_sem: RotaryEmbedding | None
    rotary_geo: RotaryEmbedding | None
    _sem_scale: float | None
    _geo_scale: float | None
    _v_head_dim: int
    decoupled_gate_logit: nn.Parameter | None
    decoupled_gate_proj: nn.Linear | None
    k_sem_null: nn.Parameter | None
    k_geo_null: nn.Parameter | None
    v_null: nn.Parameter | None
    mem_k_proj_sem: nn.Module | None
    mem_k_proj_geo: nn.Module | None
    mem_v_proj_dba: nn.Module | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        if config.mode != AttentionMode.DECOUPLED:
            raise ValueError("DecoupledAttentionLayer requires mode=decoupled")
        super().__init__(config)
        self.init_decoupled(config)
        self._init_common_modules()
        self._viz = DecoupledAttentionViz()
        self._decode = DecoupledDecode()
        self._chunked = DecoupledSDPAChunked()

    def _init_common_modules(self) -> None:
        super()._init_common_modules()
        self._init_memory_summarizer_decoupled()

    def _decoupled_gate(self, x: Tensor) -> Tensor | None:
        """Compute semantic/geometric mixing gate

        A per-head gate lets some heads lean semantic while others lean
        geometric, which encourages specialization without hard-coding a split.
        """
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(dtype=torch.float32, device=x.device)
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1).to(torch.float32)
            gate_logit = gate_bias + dyn
        return torch.sigmoid(gate_logit).to(dtype=x.dtype)

    def _null_kv_tensors(self, *, B: int, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """Build null-attention KV tensors

        A learned “null” token gives the model a safe option when no real key
        should be attended to, which can reduce pathological attention patterns.
        """
        if self.k_sem_null is None or self.k_geo_null is None or self.v_null is None:
            raise RuntimeError("null_attn enabled but null parameters are missing")
        ksn = self.k_sem_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        kgn = self.k_geo_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        vn = self.v_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        return ksn, kgn, vn

    def _forward_decoupled(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "DecoupledLayerKVCache | None",
        pos_offset: int,
        ctx: object | None = None,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
        decode_block_override: int | None = None,
    ) -> tuple[Tensor, "DecoupledLayerKVCache | None"]:
        """Compute decoupled attention output

        The semantic and geometric scores are computed separately, then combined
        (optionally with a learned gate) before mixing values.
        """
        B, T, _ = x.shape
        if self.q_sem is None or self.k_sem is None or self.q_geo is None or self.k_geo is None:
            raise RuntimeError("Decoupled mode projections not initialized")
        if self._sem_scale is None or self._geo_scale is None:
            raise RuntimeError("Decoupled mode scales not initialized")

        sem_head_dim = self.config.sem_head_dim
        geo_head_dim = self.config.geo_head_dim
        v_head_dim = int(self._v_head_dim)
        if sem_head_dim is None or geo_head_dim is None:
            raise RuntimeError("Head dims not set")

        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        qsh = self._shape(q_sem, int(sem_head_dim))
        ksh = self._shape(k_sem, int(sem_head_dim))
        if self.rotary_sem is not None:
            qsh = self.rotary_sem.rotate(qsh, pos_offset)
            ksh = self.rotary_sem.rotate(ksh, pos_offset)

        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        qgh = self._shape(q_geo, int(geo_head_dim))
        kgh = self._shape(k_geo, int(geo_head_dim))
        if self.rotary_geo is not None:
            qgh = self.rotary_geo.rotate(qgh, pos_offset)
            kgh = self.rotary_geo.rotate(kgh, pos_offset)

        v = self.v_proj(x)
        vh = self._shape(v, int(v_head_dim))

        qsh = self._apply_logit_scale(qsh)
        qgh = self._apply_logit_scale(qgh)

        g = self._decoupled_gate(x)
        if g is not None:
            qsh = qsh * (2.0 * g)
            qgh = qgh * (2.0 - 2.0 * g)

        cache_pos = None
        if cache is not None:
            cache_pos = int(cache.pos)
            old_len = int(cache.pos)
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            is_single_step_decode = (
                (not self.training)
                and old_len > 0
                and int(T) == 1
                and mask is None
                and (local_window_override is None and self.config.local_window is None)
                and x.device.type in ("cuda", "mps")
            )
            if is_single_step_decode:
                decode_block = int(decode_block_override) if decode_block_override is not None else 1024
                null_enabled = bool(getattr(self.config, "null_attn", False))
                out_fused = self._decode.run(
                    q_sem=qsh,
                    q_geo=qgh,
                    cache=cache,
                    n_heads=int(self.n_heads),
                    sem_head_dim=int(sem_head_dim),
                    geo_head_dim=int(geo_head_dim),
                    v_head_dim=int(v_head_dim),
                    sem_scale=float(self._sem_scale),
                    geo_scale=float(self._geo_scale),
                    decode_block=int(decode_block),
                    null_enabled=bool(null_enabled),
                    null_kv=self._null_kv_tensors,
                )
                y = self.out_proj(self._merge(out_fused))
                self._viz.record_activation_sample(ctx=ctx, layer=self, y=y)
                return y, cache
            if old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, int(sem_head_dim))
                kgh = self._shape(k_geo_all, int(geo_head_dim))
                vh = self._shape(v_all, int(v_head_dim))

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = local_window_override if local_window_override is not None else self.config.local_window

        sem_scale = float(self._sem_scale)
        geo_scale = float(self._geo_scale)
        dropout_p = float(self.config.dropout_p) if self.training else 0.0
        null_enabled = bool(getattr(self.config, "null_attn", False))

        dba_backend = str(getattr(self.config, "dba_train_backend", "auto") or "auto").lower().strip()
        force_sdpa = (dba_backend == "sdpa")
        if (
            (not force_sdpa)
            and q_chunk is None
            and local_window is None
            and mask is None
            and cache is None
            and x.device.type == "cuda"
            and self.training
            and not null_enabled
        ):
            out = DecoupledAttentionTraining().run(
                q_sem=qsh,
                q_geo=qgh,
                k_sem=ksh,
                k_geo=kgh,
                v=vh,
                causal=bool(self.config.is_causal) and int(T) > 1,
                sem_scale=float(sem_scale),
                geo_scale=float(geo_scale),
                dropout_p=float(dropout_p),
            )
        elif q_chunk is None and local_window is None:
            q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
            k_cat = torch.cat([ksh, kgh], dim=-1)
            if null_enabled:
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=x.dtype, device=x.device)
                k_null = torch.cat([ksn, kgn], dim=-1)
                k_cat = torch.cat([k_null.to(dtype=k_cat.dtype), k_cat], dim=2)
                vh = torch.cat([vn.to(dtype=vh.dtype), vh], dim=2)
                if mask is not None:
                    keep_null = torch.ones((*mask.shape[:-1], 1), device=mask.device, dtype=torch.bool)
                    mask = torch.cat([keep_null, mask], dim=-1)

            can_use_is_causal = (mask is None) and (cache is None) and (not null_enabled) and bool(self.config.is_causal) and int(T) > 1 and x.device.type == "cuda"
            attn_mask = mask
            is_causal = bool(can_use_is_causal)
            if attn_mask is None and (not is_causal) and bool(self.config.is_causal) and int(T) > 1:
                causal = torch.tril(torch.ones(int(T), int(k_cat.size(2)), device=x.device, dtype=torch.bool))
                if null_enabled:
                    causal[:, 0] = True
                attn_mask = causal.view(1, 1, int(T), int(k_cat.size(2)))

            out = F.scaled_dot_product_attention(
                q_cat,
                k_cat,
                vh,
                attn_mask=attn_mask,
                dropout_p=float(dropout_p),
                is_causal=bool(is_causal),
                scale=1.0,
            )
            self._viz.record_attention_matrix(ctx=ctx, layer=self, q_cat=q_cat, k_cat=k_cat)
        else:
            out = self._chunked.run(
                qsh=qsh,
                ksh=ksh,
                qgh=qgh,
                kgh=kgh,
                vh=vh,
                is_causal=bool(self.config.is_causal),
                mask=mask,
                cache_pos=cache_pos,
                q_chunk=int(q_chunk) if q_chunk is not None else int(T),
                local_window=int(local_window) if local_window is not None else None,
                sem_scale=float(sem_scale),
                geo_scale=float(geo_scale),
                dropout_p=float(dropout_p),
                null_enabled=bool(null_enabled),
                null_kv=self._null_kv_tensors,
                maybe_summarize_decoupled=self.maybe_summarize_kv_decoupled,
            )

        y = self.out_proj(self._merge(out))
        self._viz.record_activation_sample(ctx=ctx, layer=self, y=y)
        return y, cache

