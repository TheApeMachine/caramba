"""Decoupled (DBA) attention implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from carmath import neg_inf
from config.layer import AttentionLayerConfig, AttentionMode
from console import logger
from layer.attention.base import AttentionBase, SEM_ROPE_EVEN_DIM_ERROR
from layer.rope import RotaryEmbedding

if TYPE_CHECKING:
    from cache.decoupled import DecoupledLayerKVCache


# Debug aid: avoid spamming logs on every decode step.
_LOGGED_METAL_FUSED_DECODE = False
_LOGGED_FUSED_DECODE_FAILURE = False


class DecoupledAttentionLayer(AttentionBase):
    """DBA attention layer: semantic + geometric paths."""

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
    # DBA-specific memory summarization modules (dimension-correct per path).
    mem_k_proj_sem: nn.Module | None
    mem_k_proj_geo: nn.Module | None
    mem_v_proj_dba: nn.Module | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        if config.mode != AttentionMode.DECOUPLED:
            raise ValueError("DecoupledAttentionLayer requires mode=decoupled")
        super().__init__(config)
        self._init_decoupled(config)
        self._init_common_modules()

    def _init_memory_summarizer(self) -> None:
        """Initialize optional modules for DBA mem_block summarization.

        Unlike standard attention, DBA's key/value dimensions can differ across:
        - semantic keys (sem_head_dim)
        - geometric keys (geo_head_dim)
        - values (v_head_dim)

        We therefore keep separate summarization modules per path to avoid
        silent dimension mismatches.
        """
        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        sem_head_dim = int(self.config.sem_head_dim or 0)
        geo_head_dim = int(self.config.geo_head_dim or 0)
        v_head_dim = int(getattr(self, "_v_head_dim", 0) or 0)

        if kind == "linear":
            if sem_head_dim <= 0 or geo_head_dim <= 0 or v_head_dim <= 0:
                self.mem_k_proj_sem = None
                self.mem_k_proj_geo = None
                self.mem_v_proj_dba = None
                return
            self.mem_k_proj_sem = nn.Linear(sem_head_dim, sem_head_dim, bias=False)
            self.mem_k_proj_geo = nn.Linear(geo_head_dim, geo_head_dim, bias=False)
            self.mem_v_proj_dba = nn.Linear(v_head_dim, v_head_dim, bias=False)
            nn.init.eye_(cast(nn.Linear, self.mem_k_proj_sem).weight)
            nn.init.eye_(cast(nn.Linear, self.mem_k_proj_geo).weight)
            nn.init.eye_(cast(nn.Linear, self.mem_v_proj_dba).weight)
        elif kind == "conv":
            if sem_head_dim <= 0 or geo_head_dim <= 0 or v_head_dim <= 0:
                self.mem_k_proj_sem = None
                self.mem_k_proj_geo = None
                self.mem_v_proj_dba = None
                return
            self.mem_k_proj_sem = nn.Conv1d(
                sem_head_dim, sem_head_dim, kernel_size=3, padding=1, groups=sem_head_dim, bias=False
            )
            self.mem_k_proj_geo = nn.Conv1d(
                geo_head_dim, geo_head_dim, kernel_size=3, padding=1, groups=geo_head_dim, bias=False
            )
            self.mem_v_proj_dba = nn.Conv1d(
                v_head_dim, v_head_dim, kernel_size=3, padding=1, groups=v_head_dim, bias=False
            )
            # Initialize to [0.25, 0.5, 0.25] per channel.
            w = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
            for m in (self.mem_k_proj_sem, self.mem_k_proj_geo, self.mem_v_proj_dba):
                ww = cast(nn.Conv1d, m).weight
                d = int(ww.shape[0])
                ww.data.zero_()
                ww.data[:, 0, :].copy_(w.to(device=ww.device, dtype=ww.dtype).view(1, 3).expand(d, 3))
        else:
            self.mem_k_proj_sem = None
            self.mem_k_proj_geo = None
            self.mem_v_proj_dba = None

        # Ensure the standard summarizer modules aren't accidentally used in DBA.
        self.mem_k_proj = None
        self.mem_v_proj = None

    def _maybe_summarize_kv_decoupled(
        self,
        *,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        k_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Summarize older KV blocks for DBA, keeping paths aligned."""
        mem_block = getattr(self.config, "mem_block", None)
        if mem_block is None:
            return k_sem, k_geo, v, k_pos
        mb = int(mem_block)
        if mb <= 0:
            return k_sem, k_geo, v, k_pos

        if k_sem.size(2) == 0:
            return k_sem, k_geo, v, k_pos

        threshold = getattr(self.config, "mem_activation_threshold", None)
        if threshold is not None and int(k_sem.size(2)) < int(threshold):
            return k_sem, k_geo, v, k_pos

        local_window = getattr(self.config, "local_window", None)
        lw = int(local_window) if local_window is not None else 0
        T = int(k_sem.size(2))
        if lw <= 0 or lw >= T:
            return k_sem, k_geo, v, k_pos

        remote_len = T - lw
        if remote_len <= 0:
            return k_sem, k_geo, v, k_pos

        ks_r, ks_l = k_sem[:, :, :remote_len, :], k_sem[:, :, remote_len:, :]
        kg_r, kg_l = k_geo[:, :, :remote_len, :], k_geo[:, :, remote_len:, :]
        v_r, v_l = v[:, :, :remote_len, :], v[:, :, remote_len:, :]
        pos_r, pos_l = k_pos[:remote_len], k_pos[remote_len:]

        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "conv":
            # Depthwise conv over sequence dimension, per head.
            BH = int(ks_r.size(0) * ks_r.size(1))

            def _conv1d_path(x: Tensor, mod: nn.Module | None) -> Tensor:
                if mod is None:
                    return x
                d = int(x.size(-1))
                xin = x.reshape(BH, remote_len, d).transpose(1, 2)  # (BH, D, T)
                y = cast(nn.Conv1d, mod)(xin).transpose(1, 2).reshape_as(x)
                return y

            ks_r = _conv1d_path(ks_r, self.mem_k_proj_sem)
            kg_r = _conv1d_path(kg_r, self.mem_k_proj_geo)
            v_r = _conv1d_path(v_r, self.mem_v_proj_dba)

        # Pool remote into blocks.
        B0, H0, _Tr, _Dks = ks_r.shape
        n_full = remote_len // mb
        rem = remote_len - n_full * mb

        def _pool_full(x: Tensor) -> Tensor:
            if n_full > 0:
                return x[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, x.size(-1)).mean(dim=3)
            return x.new_empty((B0, H0, 0, x.size(-1)))

        ks_full = _pool_full(ks_r)
        kg_full = _pool_full(kg_r)
        v_full = _pool_full(v_r)
        pos_full = pos_r[(mb - 1) : (n_full * mb) : mb] if n_full > 0 else pos_r[:0]

        if rem > 0:
            ks_tail = ks_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            kg_tail = kg_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            v_tail = v_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            pos_tail = pos_r[remote_len - 1 : remote_len]

            ks_mem = torch.cat([ks_full, ks_tail], dim=2)
            kg_mem = torch.cat([kg_full, kg_tail], dim=2)
            v_mem = torch.cat([v_full, v_tail], dim=2)
            pos_mem = torch.cat([pos_full, pos_tail], dim=0)
        else:
            ks_mem, kg_mem, v_mem, pos_mem = ks_full, kg_full, v_full, pos_full

        if kind == "linear":
            if self.mem_k_proj_sem is not None:
                ks_mem = cast(nn.Linear, self.mem_k_proj_sem)(ks_mem)
            if self.mem_k_proj_geo is not None:
                kg_mem = cast(nn.Linear, self.mem_k_proj_geo)(kg_mem)
            if self.mem_v_proj_dba is not None:
                v_mem = cast(nn.Linear, self.mem_v_proj_dba)(v_mem)

        ks2 = torch.cat([ks_mem, ks_l], dim=2)
        kg2 = torch.cat([kg_mem, kg_l], dim=2)
        v2 = torch.cat([v_mem, v_l], dim=2)
        pos2 = torch.cat([pos_mem, pos_l], dim=0)
        return ks2, kg2, v2, pos2

    def _init_decoupled(self, config: AttentionLayerConfig) -> None:
        """Set up projections for DBA attention."""
        d_model = int(config.d_model)

        if config.sem_dim is None or config.geo_dim is None:
            raise ValueError("Decoupled mode requires sem_dim and geo_dim")

        sem_dim = int(config.sem_dim)
        geo_dim = int(config.geo_dim)
        v_dim = int(config.v_dim)

        sem_head_dim = config.sem_head_dim
        geo_head_dim = config.geo_head_dim
        if sem_head_dim is None or geo_head_dim is None:
            raise ValueError("Could not compute sem/geo head dims")
        sem_head_dim = int(sem_head_dim)
        geo_head_dim = int(geo_head_dim)

        # Semantic projections (content similarity; RoPE optional via config).
        self.q_sem = nn.Linear(d_model, sem_dim, bias=bool(config.bias))
        if bool(getattr(config, "tie_qk", False)):
            self.k_sem = self.q_sem
        else:
            self.k_sem = nn.Linear(d_model, sem_dim, bias=bool(config.bias))

        # Geometric projections (position patterns, RoPE applied)
        self.q_geo = nn.Linear(d_model, geo_dim, bias=bool(config.bias))
        self.k_geo = nn.Linear(d_model, geo_dim, bias=bool(config.bias))

        self.v_proj = nn.Linear(d_model, v_dim, bias=bool(config.bias))
        self.out_proj = nn.Linear(v_dim, d_model, bias=bool(config.bias))

        if bool(config.rope_enabled):
            if geo_head_dim % 2 != 0:
                raise ValueError("Decoupled mode with RoPE requires even geo_head_dim")
            self.rotary_geo = RotaryEmbedding(
                geo_head_dim, base=float(config.rope_base), rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary_geo = None

        # Optional RoPE on semantic path (ablation).
        if bool(config.rope_enabled) and bool(getattr(config, "rope_semantic", False)):
            if sem_head_dim % 2 != 0:
                raise ValueError(SEM_ROPE_EVEN_DIM_ERROR)
            self.rotary_sem = RotaryEmbedding(
                sem_head_dim, base=float(config.rope_base), rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary_sem = None

        self._sem_scale = 1.0 / math.sqrt(float(sem_head_dim))
        self._geo_scale = 1.0 / math.sqrt(float(geo_head_dim))
        self._v_head_dim = int(v_dim) // int(self.n_heads)

        # Optional learned null token (sink token) for DBA (ablation).
        if bool(getattr(config, "null_attn", False)):
            H = int(self.n_heads)
            v_head_dim = int(self._v_head_dim)
            self.k_sem_null = nn.Parameter(torch.zeros((H, int(sem_head_dim))))
            self.k_geo_null = nn.Parameter(torch.zeros((H, int(geo_head_dim))))
            self.v_null = nn.Parameter(torch.zeros((H, int(v_head_dim))))
            nn.init.normal_(self.k_sem_null, mean=0.0, std=0.02)
            nn.init.normal_(self.k_geo_null, mean=0.0, std=0.02)
            nn.init.normal_(self.v_null, mean=0.0, std=0.02)
        else:
            self.k_sem_null = None
            self.k_geo_null = None
            self.v_null = None

        # Optional learned gate between semantic and geometric paths
        if bool(config.decoupled_gate):
            self.decoupled_gate_logit = nn.Parameter(torch.zeros(self.n_heads))
            if bool(config.decoupled_gate_dynamic):
                self.decoupled_gate_proj = nn.Linear(d_model, self.n_heads, bias=False)
                nn.init.zeros_(self.decoupled_gate_proj.weight)
            else:
                self.decoupled_gate_proj = None
        else:
            self.decoupled_gate_logit = None
            self.decoupled_gate_proj = None

        # Standard-only attributes (kept for backwards compatibility / tests)
        self.q_proj = None
        self.k_proj = None
        self.rotary = None
        self._scale = None

    def _decoupled_gate(self, x: Tensor) -> Tensor | None:
        """Compute per-head semantic/geometric mixing weights."""
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(dtype=torch.float32, device=x.device)
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1).to(torch.float32)
            gate_logit = gate_bias + dyn
        return torch.sigmoid(gate_logit).to(dtype=x.dtype)

    def _null_kv_tensors(
        self,
        *,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return expanded (k_sem_null, k_geo_null, v_null) for null attention."""
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
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
        decode_block_override: int | None = None,
    ) -> tuple[Tensor, "DecoupledLayerKVCache | None"]:
        """DBA attention: (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V."""
        global _LOGGED_FUSED_DECODE_FAILURE
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        if self.q_sem is None or self.k_sem is None:
            raise RuntimeError("Decoupled mode projections not initialized")
        if self.q_geo is None or self.k_geo is None:
            raise RuntimeError("Decoupled mode projections not initialized")
        if self._sem_scale is None or self._geo_scale is None:
            raise RuntimeError("Decoupled mode scales not initialized")

        sem_head_dim = self.config.sem_head_dim
        geo_head_dim = self.config.geo_head_dim
        v_head_dim = self._v_head_dim
        if sem_head_dim is None or geo_head_dim is None:
            raise RuntimeError("Head dims not set")

        # Semantic path
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        qsh = self._shape(q_sem, int(sem_head_dim))
        ksh = self._shape(k_sem, int(sem_head_dim))

        if self.rotary_sem is not None:
            qsh = self.rotary_sem.rotate(qsh, pos_offset)
            ksh = self.rotary_sem.rotate(ksh, pos_offset)

        # Geometric path
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

        # Apply learned gating between paths
        g = self._decoupled_gate(x)
        if g is not None:
            qsh = qsh * (2.0 * g)
            qgh = qgh * (2.0 - 2.0 * g)

        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))

            # Fast-path: fused decode for decoupled caches.
            if (
                (not self.training)
                and old_len > 0
                and int(T) == 1
                and mask is None
                and (local_window_override is None and self.config.local_window is None)
                and x.device.type in ("cuda", "mps")
            ):
                if x.device.type == "cuda":
                    try:
                        from optimizer.fused_attention import (
                            fused_decode_available,
                            fused_decode_decoupled_q4q8q4,
                            fused_decode_decoupled_q4q8q4_2pass,
                        )

                        if fused_decode_available(cache, x.device.type):
                            decode_block = int(decode_block_override) if decode_block_override is not None else 1024
                            ksn = kgn = vn = None
                            if bool(getattr(self.config, "null_attn", False)):
                                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=x.device)
                            cache_len = int(cache.pos)
                            if cache_len > 4 * int(decode_block):
                                out_fused = fused_decode_decoupled_q4q8q4_2pass(
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
                                    k_sem_null=ksn,
                                    k_geo_null=kgn,
                                    v_null=vn,
                                )
                            else:
                                out_fused = fused_decode_decoupled_q4q8q4(
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
                                    k_sem_null=ksn,
                                    k_geo_null=kgn,
                                    v_null=vn,
                                )
                            y = self.out_proj(self._merge(out_fused))
                            return y, cache
                    except Exception as e:
                        if (not _LOGGED_FUSED_DECODE_FAILURE) and bool(getattr(self.config, "debug_fused_decode", False)):
                            logger.warning(f"Fused DBA decode unavailable; falling back: {type(e).__name__}: {e}")
                            _LOGGED_FUSED_DECODE_FAILURE = True
                elif x.device.type == "mps":
                    global _LOGGED_METAL_FUSED_DECODE
                    try:
                        from optimizer.fused_attention import (
                            fused_decode_available,
                            fused_decode_decoupled_q4q8q4,
                        )

                        if fused_decode_available(cache, x.device.type):
                            if not _LOGGED_METAL_FUSED_DECODE:
                                logger.info("Using Metal fused decoupled decode (best-effort)")
                                _LOGGED_METAL_FUSED_DECODE = True
                            ksn = kgn = vn = None
                            if bool(getattr(self.config, "null_attn", False)):
                                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=x.device)
                            out_fused = fused_decode_decoupled_q4q8q4(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                n_heads=int(self.n_heads),
                                sem_head_dim=int(sem_head_dim),
                                geo_head_dim=int(geo_head_dim),
                                v_head_dim=int(v_head_dim),
                                sem_scale=float(self._sem_scale),
                                geo_scale=float(self._geo_scale),
                                decode_block=1024,
                                k_sem_null=ksn,
                                k_geo_null=kgn,
                                v_null=vn,
                            )
                            y = self.out_proj(self._merge(out_fused))
                            return y, cache
                    except Exception as e:
                        if (not _LOGGED_FUSED_DECODE_FAILURE) and bool(getattr(self.config, "debug_fused_decode", False)):
                            logger.warning(f"Fused DBA decode unavailable; falling back: {type(e).__name__}: {e}")
                            _LOGGED_FUSED_DECODE_FAILURE = True

            if old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, int(sem_head_dim))
                kgh = self._shape(k_geo_all, int(geo_head_dim))
                vh = self._shape(v_all, int(v_head_dim))

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override if local_window_override is not None else self.config.local_window
        )

        # If no chunking/windowing, use direct attention math; else use chunked path.
        if q_chunk is None and local_window is None:
            sem_scores = torch.matmul(qsh, ksh.transpose(-2, -1)) * float(self._sem_scale)
            geo_scores = torch.matmul(qgh, kgh.transpose(-2, -1)) * float(self._geo_scale)
            scores = sem_scores + geo_scores

            if bool(getattr(self.config, "null_attn", False)):
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=x.dtype, device=x.device)
                score_null = (
                    (qsh * ksn).sum(dim=-1, keepdim=True) * float(self._sem_scale)
                    + (qgh * kgn).sum(dim=-1, keepdim=True) * float(self._geo_scale)
                )
                scores = torch.cat([score_null.to(dtype=scores.dtype), scores], dim=-1)
                vh = torch.cat([vn.to(dtype=vh.dtype), vh], dim=2)
                if mask is not None:
                    keep_null = torch.ones((*mask.shape[:-1], 1), device=mask.device, dtype=torch.bool)
                    mask = torch.cat([keep_null, mask], dim=-1)

            # Apply masking (mask semantics: True = keep)
            if mask is not None:
                scores = scores.masked_fill(~mask, ninfty)
            elif bool(self.config.is_causal) and int(T) > 1 and cache is None:
                causal = torch.tril(torch.ones(int(T), int(T), device=x.device, dtype=torch.bool))
                if bool(getattr(self.config, "null_attn", False)):
                    causal = torch.cat(
                        [torch.ones((int(T), 1), device=x.device, dtype=torch.bool), causal],
                        dim=1,
                    )
                scores = scores.masked_fill(~causal.view(1, 1, int(T), -1), ninfty)
            elif bool(self.config.is_causal) and cache is not None:
                cache_len = int(ksh.size(2))
                key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
                q_pos = (cache.pos - int(T) + torch.arange(int(T), device=x.device)).view(1, 1, int(T), 1)
                keep = key_pos <= q_pos
                if bool(getattr(self.config, "null_attn", False)):
                    keep_null = torch.ones((1, 1, int(T), 1), device=x.device, dtype=torch.bool)
                    keep = torch.cat([keep_null, keep], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
            attn = self.dropout(attn)
            out = torch.matmul(attn, vh)
        else:
            out = self._decoupled_attention_chunked(
                qsh=qsh,
                ksh=ksh,
                qgh=qgh,
                kgh=kgh,
                vh=vh,
                ninfty=ninfty,
                mask=mask,
                cache=cache,
                q_chunk=int(q_chunk) if q_chunk is not None else int(T),
                local_window=int(local_window) if local_window is not None else None,
            )

        y = self.out_proj(self._merge(out))
        return y, cache

    def _decoupled_attention_chunked(
        self,
        *,
        qsh: Tensor,
        ksh: Tensor,
        qgh: Tensor,
        kgh: Tensor,
        vh: Tensor,
        ninfty: float,
        mask: Tensor | None,
        cache: "DecoupledLayerKVCache | None",
        q_chunk: int,
        local_window: int | None,
    ) -> Tensor:
        """Chunked DBA attention to reduce peak memory for long sequences."""
        B, _H, T, _ = qsh.shape
        kT = int(ksh.size(2))
        q_chunk = max(1, int(q_chunk))

        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(int(T), device=qsh.device)
            k_pos_full = torch.arange(int(kT), device=qsh.device)
        else:
            base_q = 0
            q_pos_full = torch.arange(int(T), device=qsh.device)
            k_pos_full = torch.arange(int(kT), device=qsh.device)

        sem_scale = float(self._sem_scale) if self._sem_scale is not None else 1.0
        geo_scale = float(self._geo_scale) if self._geo_scale is not None else 1.0

        outs: list[Tensor] = []
        for i0 in range(0, int(T), int(q_chunk)):
            i1 = min(int(T), i0 + int(q_chunk))
            q_pos = q_pos_full[i0:i1]

            # Key slice range when local_window is set.
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

            k_pos = k_pos_full[k0:k1]
            q_slice_sem = qsh[:, :, i0:i1, :]
            q_slice_geo = qgh[:, :, i0:i1, :]
            k_slice_sem = ksh[:, :, k0:k1, :]
            k_slice_geo = kgh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Optional memory summarization (applied consistently across K/V).
            k_slice_sem, k_slice_geo, v_slice, k_pos = self._maybe_summarize_kv_decoupled(
                k_sem=k_slice_sem,
                k_geo=k_slice_geo,
                v=v_slice,
                k_pos=k_pos,
            )

            null_enabled = bool(getattr(self.config, "null_attn", False))
            if null_enabled:
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=qsh.device)
                k_slice_sem = torch.cat([ksn, k_slice_sem], dim=2)
                k_slice_geo = torch.cat([kgn, k_slice_geo], dim=2)
                v_slice = torch.cat([vn.to(dtype=v_slice.dtype), v_slice], dim=2)

            dropout_p = float(self.config.dropout_p) if self.training else 0.0

            if mask is None:
                attn_mask = None
                if bool(self.config.is_causal) or local_window is not None:
                    keep_tokens = torch.ones((q_pos.numel(), k_pos.numel()), device=qsh.device, dtype=torch.bool)
                    if bool(self.config.is_causal):
                        keep_tokens &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                    if local_window is not None:
                        w = int(local_window)
                        if w > 0:
                            keep_tokens &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                            if not bool(self.config.is_causal):
                                keep_tokens &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                    if null_enabled:
                        keep_null = torch.ones((q_pos.numel(), 1), device=qsh.device, dtype=torch.bool)
                        keep = torch.cat([keep_null, keep_tokens], dim=1)
                    else:
                        keep = keep_tokens
                    attn_mask = keep  # True = allowed (SDPA boolean semantics)

                q_cat = torch.cat([q_slice_sem * float(sem_scale), q_slice_geo * float(geo_scale)], dim=-1)
                k_cat = torch.cat([k_slice_sem, k_slice_geo], dim=-1)
                out = F.scaled_dot_product_attention(
                    q_cat,
                    k_cat,
                    v_slice,
                    attn_mask=attn_mask,
                    dropout_p=float(dropout_p),
                    is_causal=False,
                    scale=1.0,
                )
            else:
                # Fallback: preserve existing mask slicing semantics (True=keep).
                sem_scores = torch.matmul(q_slice_sem, k_slice_sem.transpose(-2, -1)) * sem_scale
                geo_scores = torch.matmul(q_slice_geo, k_slice_geo.transpose(-2, -1)) * geo_scale
                scores = sem_scores + geo_scores
                try:
                    m = mask[..., i0:i1, k0:k1]
                    if null_enabled:
                        keep_null = torch.ones((*m.shape[:-1], 1), device=m.device, dtype=torch.bool)
                        m = torch.cat([keep_null, m], dim=-1)
                    scores = scores.masked_fill(~m, ninfty)
                except Exception:
                    logger.warning("Mask slice/masked_fill failed; continuing without extra mask")
                attn = F.softmax(scores.float(), dim=-1).to(qsh.dtype)
                attn = self.dropout(attn)
                out = torch.matmul(attn, v_slice)
            outs.append(out)

        return torch.cat(outs, dim=2)
