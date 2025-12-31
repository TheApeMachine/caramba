"""Standard and GQA attention implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.layer.attention.base import AttentionBase
from caramba.layer.rope import RotaryEmbedding

if TYPE_CHECKING:
    from caramba.cache.layer import LayerKVCache


class StandardAttentionLayer(AttentionBase):
    """Multi-head attention (standard + GQA) implementation."""

    q_proj: nn.Linear | None
    k_proj: nn.Linear | None
    v_proj: nn.Linear
    out_proj: nn.Linear
    rotary: RotaryEmbedding | None
    _scale: float | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        if config.mode == AttentionMode.DECOUPLED:
            raise ValueError("StandardAttentionLayer cannot be constructed with mode=decoupled")
        super().__init__(config)
        self._init_standard(config)
        self._init_common_modules()

    def _init_standard(self, config: AttentionLayerConfig) -> None:
        """Set up projections for standard/GQA attention."""
        d_model = int(config.d_model)
        attn_dim = int(config.attn_dim) if config.attn_dim else d_model
        kv_dim = int(self.n_kv_heads) * int(self.head_dim)

        self.q_proj = nn.Linear(d_model, attn_dim, bias=bool(config.bias))
        self.k_proj = nn.Linear(d_model, kv_dim, bias=bool(config.bias))
        self.v_proj = nn.Linear(d_model, kv_dim, bias=bool(config.bias))
        self.out_proj = nn.Linear(attn_dim, d_model, bias=bool(config.bias))

        if bool(config.rope_enabled):
            self.rotary = RotaryEmbedding(
                int(self.head_dim), base=float(config.rope_base), rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary = None

        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        # Decoupled-only attributes (kept for backwards compatibility / tests)
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

    def _forward_standard(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "LayerKVCache | None",
        pos_offset: int,
        ctx: object | None = None,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
    ) -> tuple[Tensor, "LayerKVCache | None"]:
        """Standard/GQA attention: Q·K^T → softmax → V."""
        _B, T, _ = x.shape

        if self.q_proj is None or self.k_proj is None:
            raise RuntimeError("Standard mode projections not initialized")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self._shape(q, self.head_dim, self.n_heads)
        kh = self._shape(k, self.head_dim, self.n_kv_heads)
        vh = self._shape(v, self.head_dim, self.n_kv_heads)

        if self.rotary is not None:
            qh = self.rotary.rotate(qh, pos_offset)
            kh = self.rotary.rotate(kh, pos_offset)

        qh = self._apply_logit_scale(qh)

        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(kh), self._merge(vh))
            if old_len > 0:
                k_all, v_all = cache.get(dtype=qh.dtype)
                kh = self._shape(k_all, self.head_dim, self.n_kv_heads)
                vh = self._shape(v_all, self.head_dim, self.n_kv_heads)

        if self.group_size > 1:
            kh = kh.repeat_interleave(self.group_size, dim=1)
            vh = vh.repeat_interleave(self.group_size, dim=1)

        # --- Training viz: compute small attention heatmaps (best-effort) ---
        #
        # SDPA does not return attention weights, so we recompute a tiny attention
        # matrix for a few heads/tokens only (cheap).
        try:
            from caramba.instrumentation.viz import TrainingVizContext

            if ctx is not None and isinstance(ctx, TrainingVizContext) and ctx.enabled:
                idx = int(getattr(self, "_viz_index", -1))
                name = str(getattr(self, "_viz_name", ""))
                mode = str(getattr(getattr(self, "mode", None), "value", getattr(self, "mode", "")))
                if idx >= 0:
                    tq = int(min(int(ctx.max_tokens), int(qh.size(2))))
                    tk = int(min(int(ctx.max_tokens), int(kh.size(2))))
                    hh = int(min(int(ctx.max_heads), int(qh.size(1))))
                    if tq > 0 and tk > 0 and hh > 0:
                        qs = qh[:, :hh, :tq, :].float()
                        ks = kh[:, :hh, :tk, :].float()
                        # (B, H, tq, tk)
                        logits = torch.matmul(qs, ks.transpose(-2, -1)) * float(self._scale or 1.0)
                        # Apply causal mask in training (pos_offset is typically 0).
                        if bool(self.config.is_causal):
                            causal = torch.tril(
                                torch.ones((tq, tk), device=logits.device, dtype=torch.bool)
                            )
                            logits = logits.masked_fill(~causal.view(1, 1, tq, tk), float("-inf"))
                        probs = torch.softmax(logits, dim=-1)
                        eps = 1e-9
                        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean(dim=-1)  # (B,H)
                        ent_list = [float(x) for x in ent[0].detach().cpu().tolist()]
                        mats = [probs[0, h, :, :].detach() for h in range(int(hh))]
                        ctx.record_attention_matrix(
                            idx=idx,
                            name=name,
                            mode=mode,
                            n_heads=int(getattr(self, "n_heads", hh)),
                            matrices=mats,
                            entropies=ent_list,
                        )
        except Exception:
            pass

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override if local_window_override is not None else self.config.local_window
        )
        dropout_p = float(self.config.dropout_p) if self.training else 0.0

        # Memory-efficient path: compute attention in query chunks and/or restrict to a window.
        # Also fixes causal masking for cached inference prefill/decode by using explicit masks.
        if mask is None and (q_chunk is not None or local_window is not None or cache is not None):
            out = self._sdp_attention_chunked(
                qh,
                kh,
                vh,
                pos_offset=pos_offset,
                cache=cache,
                q_chunk=int(q_chunk) if q_chunk is not None else int(T),
                local_window=int(local_window) if local_window is not None else None,
                dropout_p=float(dropout_p),
            )
        else:
            is_causal = bool(self.config.is_causal) and mask is None and int(T) > 1 and cache is None
            # MPS has historically had subtle issues with SDPA's `is_causal=True` fast-path.
            # Prefer explicit causal mask path on MPS.
            if is_causal and qh.device.type == "mps":
                out = self._sdp_attention_chunked(
                    qh,
                    kh,
                    vh,
                    pos_offset=pos_offset,
                    cache=cache,
                    q_chunk=int(T),
                    local_window=None,
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
        # Training viz: record a small activation slice from the output.
        try:
            from caramba.instrumentation.viz import TrainingVizContext

            if ctx is not None and isinstance(ctx, TrainingVizContext) and ctx.enabled:
                idx = int(getattr(self, "_viz_index", -1))
                name = str(getattr(self, "_viz_name", ""))
                mode = str(getattr(getattr(self, "mode", None), "value", getattr(self, "mode", "")))
                if idx >= 0:
                    ctx.record_activation_sample(
                        idx=idx,
                        name=name,
                        mode=mode,
                        n_heads=int(getattr(self, "n_heads", 0) or 0) or None,
                        y=y,
                    )
        except Exception:
            pass
        return y, cache

    def _sdp_attention_chunked(
        self,
        qh: Tensor,
        kh: Tensor,
        vh: Tensor,
        *,
        pos_offset: int,
        cache: "LayerKVCache | None",
        q_chunk: int,
        local_window: int | None,
        dropout_p: float,
    ) -> Tensor:
        """Scaled-dot-product attention with chunking/windowing for lower peak memory."""

        _B, _H, T, _D = qh.shape
        kT = int(kh.size(2))

        # Base positions: in cached mode, q positions are aligned to the global cache index.
        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(int(T), device=qh.device)
            k_pos_full = torch.arange(int(kT), device=qh.device)
        else:
            base_q = int(pos_offset)
            q_pos_full = base_q + torch.arange(int(T), device=qh.device)
            k_pos_full = int(pos_offset) + torch.arange(int(kT), device=qh.device)

        outs: list[Tensor] = []
        q_chunk = max(1, int(q_chunk))
        for i0 in range(0, int(T), int(q_chunk)):
            i1 = min(int(T), i0 + int(q_chunk))

            q_pos = q_pos_full[i0:i1]

            # Key range selection to reduce work further when local_window is set.
            k0 = 0
            k1 = int(kT)
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    q_min = int(base_q + i0)
                    q_max = int(base_q + i1 - 1)
                    if bool(self.config.is_causal):
                        k0 = max(0, q_min - w + 1)
                        k1 = min(int(kT), q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(int(kT), q_max + w)

            q_slice = qh[:, :, i0:i1, :]
            k_slice = kh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Build a boolean "allowed positions" matrix for SDPA.
            if bool(self.config.is_causal) or local_window is not None:
                k_pos = k_pos_full[k0:k1]
                # Optional memory summarization over the key/value sequence.
                k_slice, v_slice, k_pos = self._maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                allowed = torch.ones((q_pos.numel(), k_pos.numel()), device=qh.device, dtype=torch.bool)
                if bool(self.config.is_causal):
                    allowed &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                if local_window is not None:
                    w = int(local_window)
                    if w > 0:
                        allowed &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                        if not bool(self.config.is_causal):
                            allowed &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                attn_mask = allowed  # True = allowed for SDPA
            else:
                # Even without causal/window masks, allow optional summarization.
                k_pos = k_pos_full[k0:k1]
                k_slice, v_slice, _k_pos2 = self._maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                attn_mask=attn_mask,
                dropout_p=float(dropout_p),
                is_causal=False,
                scale=self._scale,
            )
            outs.append(out)

        return torch.cat(outs, dim=2)

