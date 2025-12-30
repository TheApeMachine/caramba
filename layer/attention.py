"""Unified attention layer supporting standard, GQA, and DBA modes.

Attention is the core mechanism that lets tokens "look at" each other.
This module supports three modes:
- standard: Full multi-head attention (every head has its own K/V)
- gqa: Grouped-query attention (fewer K/V heads, shared across Q heads)
- decoupled: DBA with separate semantic and geometric key paths

The decoupled (DBA) mode is our research contribution—it splits attention
into content-based (semantic) and position-based (geometric) components,
enabling significant KV-cache compression.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.layer import AttentionLayerConfig, AttentionMode
from console import logger
from layer.rope import RotaryEmbedding

if TYPE_CHECKING:
    from cache.decoupled import DecoupledLayerKVCache
    from cache.layer import LayerKVCache

# Error message constants (keep exact wording for tests/log searchability).
SEM_ROPE_EVEN_DIM_ERROR = "Decoupled mode with semantic RoPE requires even sem_head_dim"

# Lazy-cached reference to avoid per-call import overhead
_InferContext: type | None = None

# Debug aid: avoid spamming logs on every decode step.
_LOGGED_METAL_FUSED_DECODE = False


def _get_infer_context_type() -> type:
    """Get the InferContext type, caching it on first access."""
    global _InferContext
    if _InferContext is None:
        from infer.context import InferContext

        _InferContext = InferContext
    return _InferContext


def _neg_inf(dtype: torch.dtype) -> float:
    """Return a large negative value safe for the given dtype.

    Float16 has limited range, so we use -65504 instead of -1e9.
    """
    if dtype == torch.float16:
        return -65504.0
    return -1e9


class AttentionLayer(nn.Module):
    """Multi-head attention with standard/GQA/DBA support.

    The mode determines the attention computation:
    - standard/gqa: Traditional Q·K^T → softmax → V pipeline
    - decoupled: (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V

    DBA's key insight is that content routing (semantic) and position
    patterns (geometric) are separate concerns that can use compressed
    key projections, reducing KV-cache memory dramatically.
    """

    # Type declarations for all attributes
    q_proj: nn.Linear | None
    k_proj: nn.Linear | None
    v_proj: nn.Linear
    out_proj: nn.Linear
    q_sem: nn.Linear | None
    k_sem: nn.Linear | None
    q_geo: nn.Linear | None
    k_geo: nn.Linear | None
    rotary: RotaryEmbedding | None
    rotary_sem: RotaryEmbedding | None
    rotary_geo: RotaryEmbedding | None
    decoupled_gate_logit: nn.Parameter | None
    decoupled_gate_proj: nn.Linear | None
    logit_scale: nn.Parameter | None
    _scale: float | None
    _sem_scale: float | None
    _geo_scale: float | None
    _v_head_dim: int
    mem_k_proj: nn.Module | None
    mem_v_proj: nn.Module | None
    k_sem_null: nn.Parameter | None
    k_geo_null: nn.Parameter | None
    v_null: nn.Parameter | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        """Initialize attention based on the configured mode.

        The config specifies dimensions, number of heads, RoPE settings,
        and mode-specific parameters (sem_dim, geo_dim for DBA).
        """
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.dropout = nn.Dropout(config.dropout_p)
        self.mem_k_proj = None
        self.mem_v_proj = None

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        self.group_size = self.n_heads // self.n_kv_heads

        if self.mode == AttentionMode.DECOUPLED:
            self._init_decoupled(config)
        else:
            self._init_standard(config)

        # Optional learned temperature scaling per head
        self.logit_scale = None
        if config.learned_temp:
            self.logit_scale = nn.Parameter(torch.zeros(self.n_heads))

        # Optional long-sequence memory summarization modules.
        self._init_memory_summarizer()

    def _init_memory_summarizer(self) -> None:
        """Initialize optional modules for mem_block summarization."""

        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "linear":
            d = int(self.head_dim)
            self.mem_k_proj = nn.Linear(d, d, bias=False)
            self.mem_v_proj = nn.Linear(d, d, bias=False)
            # Initialize as identity so "linear" starts equivalent to mean pooling.
            nn.init.eye_(self.mem_k_proj.weight)
            nn.init.eye_(self.mem_v_proj.weight)
        elif kind == "conv":
            d = int(self.head_dim)
            # Depthwise conv over sequence dimension; stable and cheap.
            self.mem_k_proj = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
            self.mem_v_proj = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
            # Initialize to a simple smoothing kernel [0.25, 0.5, 0.25] per channel.
            w = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
            wk = cast(nn.Conv1d, self.mem_k_proj).weight
            wv = cast(nn.Conv1d, self.mem_v_proj).weight
            wk.data.zero_()
            wv.data.zero_()
            wk.data[:, 0, :].copy_(w.to(device=wk.device, dtype=wk.dtype).view(1, 3).expand(d, 3))
            wv.data[:, 0, :].copy_(w.to(device=wv.device, dtype=wv.dtype).view(1, 3).expand(d, 3))
        else:
            self.mem_k_proj = None
            self.mem_v_proj = None

    def _maybe_summarize_kv(
        self,
        *,
        k: Tensor,
        v: Tensor,
        k_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Summarize older KV blocks into memory tokens for long sequences.

        This is an approximation used to reduce compute/memory on long sequences.
        It is inactive unless config.mem_block is set and activation threshold
        (if any) is met.
        """

        mem_block = getattr(self.config, "mem_block", None)
        if mem_block is None:
            return k, v, k_pos
        mb = int(mem_block)
        if mb <= 0:
            return k, v, k_pos

        # Guard for empty k tensor.
        if k.size(2) == 0:
            return k, v, k_pos

        threshold = getattr(self.config, "mem_activation_threshold", None)
        if threshold is not None and int(k.size(2)) < int(threshold):
            return k, v, k_pos

        # If local_window is set, keep that many most-recent tokens uncompressed.
        local_window = getattr(self.config, "local_window", None)
        lw = int(local_window) if local_window is not None else 0
        T = int(k.size(2))
        if lw <= 0 or lw >= T:
            return k, v, k_pos

        remote_len = T - lw
        if remote_len <= 0:
            return k, v, k_pos

        k_remote = k[:, :, :remote_len, :]
        v_remote = v[:, :, :remote_len, :]
        k_local = k[:, :, remote_len:, :]
        v_local = v[:, :, remote_len:, :]
        pos_remote = k_pos[:remote_len]
        pos_local = k_pos[remote_len:]

        # Optional conv preprocessing (per head) before pooling.
        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "conv" and self.mem_k_proj is not None and self.mem_v_proj is not None:
            # (B,H,T,D) -> (B*H,D,T)
            BH = int(k_remote.size(0) * k_remote.size(1))
            d = int(k_remote.size(-1))
            k_in = k_remote.reshape(BH, remote_len, d).transpose(1, 2)
            v_in = v_remote.reshape(BH, remote_len, d).transpose(1, 2)
            k_f = cast(nn.Conv1d, self.mem_k_proj)(k_in).transpose(1, 2).reshape_as(k_remote)
            v_f = cast(nn.Conv1d, self.mem_v_proj)(v_in).transpose(1, 2).reshape_as(v_remote)
            k_remote = k_f
            v_remote = v_f

        # Pool remote into blocks (vectorized; avoid Python loops and `.item()` syncs).
        B0, H0, _Tr, D0 = k_remote.shape
        n_full = remote_len // mb
        rem = remote_len - n_full * mb

        if n_full > 0:
            k_full = k_remote[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, D0).mean(dim=3)
            v_full = v_remote[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, D0).mean(dim=3)
            pos_full = pos_remote[(mb - 1) : (n_full * mb) : mb]
        else:
            # Construct empty tensors without calling `mean` on an empty dimension.
            k_full = k_remote.new_empty((B0, H0, 0, D0))
            v_full = v_remote.new_empty((B0, H0, 0, D0))
            pos_full = pos_remote[:0]

        if rem > 0:
            k_tail = k_remote[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)  # (B,H,1,D)
            v_tail = v_remote[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            pos_tail = pos_remote[remote_len - 1 : remote_len]  # (1,)
            k_mem = torch.cat([k_full, k_tail], dim=2)
            v_mem = torch.cat([v_full, v_tail], dim=2)
            pos_mem = torch.cat([pos_full, pos_tail], dim=0)
        else:
            k_mem = k_full
            v_mem = v_full
            pos_mem = pos_full

        if kind == "linear" and self.mem_k_proj is not None and self.mem_v_proj is not None:
            k_mem = cast(nn.Linear, self.mem_k_proj)(k_mem)
            v_mem = cast(nn.Linear, self.mem_v_proj)(v_mem)

        k2 = torch.cat([k_mem, k_local], dim=2)
        v2 = torch.cat([v_mem, v_local], dim=2)
        pos2 = torch.cat([pos_mem, pos_local], dim=0)
        return k2, v2, pos2

    def _init_standard(self, config: AttentionLayerConfig) -> None:
        """Set up projections for standard/GQA attention."""
        d_model = config.d_model
        attn_dim = config.attn_dim if config.attn_dim else d_model
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, attn_dim, bias=config.bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(attn_dim, d_model, bias=config.bias)

        if config.rope_enabled:
            self.rotary = RotaryEmbedding(
                self.head_dim, base=config.rope_base, rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary = None
        self.rotary_sem = None

        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        # Decoupled projections unused in this mode
        self.q_sem = None
        self.k_sem = None
        self.q_geo = None
        self.k_geo = None
        self.rotary_geo = None
        self._sem_scale = None
        self._geo_scale = None
        self._v_head_dim = self.head_dim
        self.decoupled_gate_logit = None
        self.decoupled_gate_proj = None
        self.k_sem_null = None
        self.k_geo_null = None
        self.v_null = None

    def _init_decoupled(self, config: AttentionLayerConfig) -> None:
        """Set up projections for DBA attention.

        DBA has two key paths:
        - Semantic (no RoPE): learns content/topic similarity
        - Geometric (with RoPE): learns position-based patterns

        These are combined before softmax, allowing the model to learn
        which path to emphasize for different attention patterns.
        """
        d_model = config.d_model

        if config.sem_dim is None or config.geo_dim is None:
            raise ValueError("Decoupled mode requires sem_dim and geo_dim")

        sem_dim = config.sem_dim
        geo_dim = config.geo_dim
        v_dim = config.v_dim

        sem_head_dim = config.sem_head_dim
        geo_head_dim = config.geo_head_dim

        if sem_head_dim is None or geo_head_dim is None:
            raise ValueError("Could not compute sem/geo head dims")

        # Semantic projections (content similarity; RoPE optional via config).
        self.q_sem = nn.Linear(d_model, sem_dim, bias=config.bias)
        if bool(getattr(config, "tie_qk", False)):
            # Tie semantic Q/K to test symmetric semantics ablations.
            self.k_sem = self.q_sem
        else:
            self.k_sem = nn.Linear(d_model, sem_dim, bias=config.bias)

        # Geometric projections (position patterns, RoPE applied)
        self.q_geo = nn.Linear(d_model, geo_dim, bias=config.bias)
        self.k_geo = nn.Linear(d_model, geo_dim, bias=config.bias)

        self.v_proj = nn.Linear(d_model, v_dim, bias=config.bias)
        self.out_proj = nn.Linear(v_dim, d_model, bias=config.bias)

        if config.rope_enabled:
            if geo_head_dim % 2 != 0:
                raise ValueError("Decoupled mode with RoPE requires even geo_head_dim")
            self.rotary_geo = RotaryEmbedding(
                geo_head_dim, base=config.rope_base, rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary_geo = None

        # Optional RoPE on semantic path (ablation).
        if bool(getattr(config, "rope_semantic", False)):
            if sem_head_dim % 2 != 0:
                raise ValueError(SEM_ROPE_EVEN_DIM_ERROR)
            self.rotary_sem = RotaryEmbedding(
                sem_head_dim, base=config.rope_base, rope_scaling=getattr(config, "rope_scaling", None)
            )
        else:
            self.rotary_sem = None

        self._sem_scale = 1.0 / math.sqrt(float(sem_head_dim))
        self._geo_scale = 1.0 / math.sqrt(float(geo_head_dim))
        self._v_head_dim = v_dim // self.n_heads

        # Optional learned null token (sink token) for DBA (ablation).
        if bool(getattr(config, "null_attn", False)):
            H = int(self.n_heads)
            v_head_dim = int(self._v_head_dim)
            # Store per-head null vectors (broadcasted over batch/sequence at runtime).
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
        if config.decoupled_gate:
            self.decoupled_gate_logit = nn.Parameter(torch.zeros(self.n_heads))
            if config.decoupled_gate_dynamic:
                self.decoupled_gate_proj = nn.Linear(d_model, self.n_heads, bias=False)
                nn.init.zeros_(self.decoupled_gate_proj.weight)
            else:
                self.decoupled_gate_proj = None
        else:
            self.decoupled_gate_logit = None
            self.decoupled_gate_proj = None

        # Standard projections unused in this mode
        self.q_proj = None
        self.k_proj = None
        self.rotary = None
        self._scale = None

    def _shape(self, x: Tensor, head_dim: int, n_heads: int | None = None) -> Tensor:
        """Reshape (B, T, D) → (B, H, T, head_dim) for attention."""
        B, T, _ = x.shape
        H = n_heads if n_heads is not None else self.n_heads
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: Tensor) -> Tensor:
        """Reshape (B, H, T, head_dim) → (B, T, D) after attention."""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale(self, q: Tensor) -> Tensor:
        """Apply learned per-head temperature scaling."""
        if self.logit_scale is None:
            return q
        s = self.logit_scale.float().clamp(min=-8.0, max=8.0)
        scale = torch.exp(s).to(dtype=q.dtype).view(1, -1, 1, 1)
        return q * scale

    def _decoupled_gate(self, x: Tensor) -> Tensor | None:
        """Compute per-head semantic/geometric mixing weights.

        Returns a gate value in [0, 1] where 1 means fully semantic
        and 0 means fully geometric.
        """
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(
            dtype=torch.float32, device=x.device
        )
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = (
                self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1).to(torch.float32)
            )
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
        # Expected shape for decode paths: (B, H, 1, head_dim)
        ksn = self.k_sem_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        kgn = self.k_geo_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        vn = self.v_null.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1).to(device=device, dtype=dtype)
        return ksn, kgn, vn

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        cache: "LayerKVCache | DecoupledLayerKVCache | None" = None,
        pos_offset: int = 0,
        ctx: object | None = None,
    ) -> tuple[Tensor, "LayerKVCache | DecoupledLayerKVCache | None"]:
        """Compute attention and return output with updated cache.

        Args:
            x: Input features (B, T, d_model)
            mask: Optional attention mask (B, 1, T, S)
            cache: Optional KV cache for incremental decoding
            pos_offset: Position offset for RoPE in cached generation
            ctx: Optional InferContext containing caches and position info

        Returns:
            (output, updated_cache) tuple
        """
        InferContextType = _get_infer_context_type()
        q_chunk_override: int | None = None
        local_window_override: int | None = None
        decode_block_override: int | None = None
        if ctx is not None and isinstance(ctx, InferContextType):
            # The linter can't refine `object` via isinstance() against a runtime-loaded type.
            ictx = cast(Any, ctx)
            cache = ictx.next_cache()
            pos_offset = ictx.pos_offset
            if ictx.attn_mask is not None:
                mask = ictx.attn_mask
            q_chunk_override = getattr(ictx, "q_chunk", None)
            local_window_override = getattr(ictx, "local_window", None)
            decode_block_override = getattr(ictx, "decode_block", None)

        if self.mode == AttentionMode.DECOUPLED:
            decoupled_cache = cast("DecoupledLayerKVCache | None", cache)
            return self._forward_decoupled(
                x,
                mask=mask,
                cache=decoupled_cache,
                pos_offset=pos_offset,
                q_chunk_override=q_chunk_override,
                local_window_override=local_window_override,
                decode_block_override=decode_block_override,
            )
        standard_cache = cast("LayerKVCache | None", cache)
        return self._forward_standard(
            x,
            mask=mask,
            cache=standard_cache,
            pos_offset=pos_offset,
            q_chunk_override=q_chunk_override,
            local_window_override=local_window_override,
        )

    def _forward_standard(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "LayerKVCache | None",
        pos_offset: int,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
    ) -> tuple[Tensor, "LayerKVCache | None"]:
        """Standard/GQA attention: Q·K^T → softmax → V."""
        B, T, _ = x.shape

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

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override
            if local_window_override is not None
            else self.config.local_window
        )
        dropout_p = self.config.dropout_p if self.training else 0.0

        # Memory-efficient path: compute attention in query chunks and/or restrict to a window.
        # Also fixes causal masking for cached inference prefill/decode by using explicit masks.
        if mask is None and (q_chunk is not None or local_window is not None or cache is not None):
            out = self._sdp_attention_chunked(
                qh,
                kh,
                vh,
                pos_offset=pos_offset,
                cache=cache,
                q_chunk=int(q_chunk) if q_chunk is not None else T,
                local_window=int(local_window) if local_window is not None else None,
                dropout_p=float(dropout_p),
            )
        else:
            is_causal = self.config.is_causal and mask is None and T > 1 and cache is None
            # MPS has historically had subtle issues with SDPA's `is_causal=True` fast-path.
            # To keep teacher parity stable (especially for Llama-family models), prefer an
            # explicit causal mask path on MPS.
            if is_causal and qh.device.type == "mps":
                out = self._sdp_attention_chunked(
                    qh,
                    kh,
                    vh,
                    pos_offset=pos_offset,
                    cache=cache,
                    q_chunk=T,
                    local_window=None,
                    dropout_p=float(dropout_p),
                )
            else:
                out = F.scaled_dot_product_attention(
                    qh,
                    kh,
                    vh,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=self._scale,
                )

        y = self.out_proj(self._merge(out))
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

        B, H, T, D = qh.shape
        kT = kh.size(2)

        # Base positions: in cached mode, q positions are aligned to the global cache index.
        # We mirror the decoupled implementation semantics.
        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(T, device=qh.device)
            k_pos_full = torch.arange(kT, device=qh.device)
        else:
            base_q = int(pos_offset)
            q_pos_full = base_q + torch.arange(T, device=qh.device)
            k_pos_full = int(pos_offset) + torch.arange(kT, device=qh.device)

        outs: list[Tensor] = []
        q_chunk = max(1, int(q_chunk))
        for i0 in range(0, T, q_chunk):
            i1 = min(T, i0 + q_chunk)

            q_pos = q_pos_full[i0:i1]

            # Key range selection to reduce work further when local_window is set.
            k0 = 0
            k1 = kT
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    # q_pos is always a contiguous range derived from base_q, so we can avoid
                    # device reductions + `.item()` syncs here.
                    q_min = int(base_q + i0)
                    q_max = int(base_q + i1 - 1)
                    if self.config.is_causal:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + w)

            q_slice = qh[:, :, i0:i1, :]
            k_slice = kh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Build a boolean "allowed positions" matrix for SDPA.
            if self.config.is_causal or local_window is not None:
                k_pos = k_pos_full[k0:k1]
                # Optional memory summarization over the key/value sequence.
                k_slice, v_slice, k_pos = self._maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                allowed = torch.ones((q_pos.numel(), k_pos.numel()), device=qh.device, dtype=torch.bool)
                if self.config.is_causal:
                    allowed &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                if local_window is not None:
                    w = int(local_window)
                    if w > 0:
                        allowed &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                        if not self.config.is_causal:
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
                dropout_p=dropout_p,
                is_causal=False,
                scale=self._scale,
            )
            outs.append(out)

        return torch.cat(outs, dim=2)

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
        """DBA attention: (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V.

        The semantic path captures content-based routing (what tokens to
        attend to based on meaning). The geometric path captures position-
        based patterns (local attention, recency bias). Combining them
        before softmax lets the model learn the right balance.
        """
        B, T, _ = x.shape
        ninfty = _neg_inf(x.dtype)

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

        # Semantic path (no RoPE—pure content similarity)
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        qsh = self._shape(q_sem, sem_head_dim)
        ksh = self._shape(k_sem, sem_head_dim)

        if self.rotary_sem is not None:
            qsh = self.rotary_sem.rotate(qsh, pos_offset)
            ksh = self.rotary_sem.rotate(ksh, pos_offset)

        # Geometric path (with RoPE—position patterns)
        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        qgh = self._shape(q_geo, geo_head_dim)
        kgh = self._shape(k_geo, geo_head_dim)

        if self.rotary_geo is not None:
            qgh = self.rotary_geo.rotate(qgh, pos_offset)
            kgh = self.rotary_geo.rotate(kgh, pos_offset)

        v = self.v_proj(x)
        vh = self._shape(v, v_head_dim)

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
            # We only use this for single-token decode with no extra masking/windowing.
            if (
                (not self.training)
                and old_len > 0
                and int(T) == 1
                and mask is None
                and (local_window_override is None and self.config.local_window is None)
                # Note: q_chunk does not affect correctness for single-token decode,
                # so we still allow the fused decode fast-path even if a training
                # config sets q_chunk for long-sequence chunking.
                and x.device.type in ("cuda", "mps")
            ):
                if x.device.type == "cuda":
                    # CUDA: Triton fused decode for quantized caches.
                    try:
                        from optimizer.fused_attention import (
                            fused_decode_available,
                            fused_decode_decoupled_q4q8q4,
                            fused_decode_decoupled_q4q8q4_2pass,
                        )

                        if fused_decode_available(cache, x.device.type):
                            decode_block = (
                                int(decode_block_override)
                                if decode_block_override is not None
                                else 1024
                            )
                            ksn = kgn = vn = None
                            if bool(getattr(self.config, "null_attn", False)):
                                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=x.device)
                            # Heuristic: for very long prefixes, prefer split-K 2-pass decode.
                            cache_len = int(cache.pos)
                            if cache_len > 4 * int(decode_block):
                                try:
                                    out_fused = fused_decode_decoupled_q4q8q4_2pass(
                                        q_sem=qsh,
                                        q_geo=qgh,
                                        cache=cache,
                                        n_heads=int(self.n_heads),
                                        sem_head_dim=int(sem_head_dim),
                                        geo_head_dim=int(geo_head_dim),
                                        v_head_dim=int(v_head_dim),
                                        decode_block=int(decode_block),
                                        sem_scale=float(self._sem_scale),
                                        geo_scale=float(self._geo_scale),
                                        k_sem_null=ksn,
                                        k_geo_null=kgn,
                                        v_null=vn,
                                    )
                                except Exception:
                                    out_fused = fused_decode_decoupled_q4q8q4(
                                        q_sem=qsh,
                                        q_geo=qgh,
                                        cache=cache,
                                        n_heads=int(self.n_heads),
                                        sem_head_dim=int(sem_head_dim),
                                        geo_head_dim=int(geo_head_dim),
                                        v_head_dim=int(v_head_dim),
                                        decode_block=int(decode_block),
                                        sem_scale=float(self._sem_scale),
                                        geo_scale=float(self._geo_scale),
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
                                    decode_block=int(decode_block),
                                    sem_scale=float(self._sem_scale),
                                    geo_scale=float(self._geo_scale),
                                    k_sem_null=ksn,
                                    k_geo_null=kgn,
                                    v_null=vn,
                                )
                            y = self.out_proj(self._merge(out_fused.to(dtype=x.dtype)))
                            return y, cache
                    except Exception:
                        # Any failure in optional fused kernels should silently fall back
                        # to the safe PyTorch implementation.
                        logger.warning("Fused decode failed, falling back to PyTorch implementation")

                if x.device.type == "mps":
                    # MPS: custom Metal fused decode for fp16 caches (no quantization).
                    try:
                        from optimizer.metal import dba_decode_fp16, metal_dba_decode_available

                        # Only attempt when the underlying KV caches are fp16 buffers.
                        k_sem_buf = getattr(cache.k_sem, "buf", None)
                        k_geo_buf = getattr(cache.k_geo, "buf", None)
                        v_buf = getattr(cache.v, "buf", None)
                        if (
                            metal_dba_decode_available()
                            and getattr(cache.k_sem, "kind", None) == "fp16"
                            and getattr(cache.k_geo, "kind", None) == "fp16"
                            and getattr(cache.v, "kind", None) == "fp16"
                            and k_sem_buf is not None
                            and k_geo_buf is not None
                            and v_buf is not None
                        ):
                            global _LOGGED_METAL_FUSED_DECODE
                            if not _LOGGED_METAL_FUSED_DECODE:
                                logger.info("Using Metal fused decode")
                                _LOGGED_METAL_FUSED_DECODE = True
                            S = int(cache.pos)
                            k_sem_view = k_sem_buf.narrow(1, 0, S)
                            k_geo_view = k_geo_buf.narrow(1, 0, S)
                            v_view = v_buf.narrow(1, 0, S)
                            ksn = kgn = vn = None
                            if bool(getattr(self.config, "null_attn", False)):
                                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=x.device)
                            out_fused = dba_decode_fp16(
                                q_sem=qsh,
                                q_geo=qgh,
                                k_sem=k_sem_view,
                                k_geo=k_geo_view,
                                v=v_view,
                                sem_scale=float(self._sem_scale),
                                geo_scale=float(self._geo_scale),
                                k_sem_null=ksn,
                                k_geo_null=kgn,
                                v_null=vn,
                            )
                            y = self.out_proj(self._merge(out_fused.to(dtype=x.dtype)))
                            return y, cache
                    except Exception as e:
                        logger.warning(
                            f"Metal fused decode failed, falling back to PyTorch implementation: {e}"
                        )

            if old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, sem_head_dim)
                kgh = self._shape(k_geo_all, geo_head_dim)
                vh = self._shape(v_all, v_head_dim)

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override
            if local_window_override is not None
            else self.config.local_window
        )

        if q_chunk is None and local_window is None:
            # Combine semantic and geometric scores
            sem_scores = torch.matmul(qsh, ksh.transpose(-2, -1)) * self._sem_scale
            geo_scores = torch.matmul(qgh, kgh.transpose(-2, -1)) * self._geo_scale
            scores = sem_scores + geo_scores

            if bool(getattr(self.config, "null_attn", False)):
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=x.dtype, device=x.device)
                # score_null: (B,H,T,1)
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
            elif self.config.is_causal and T > 1 and cache is None:
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                if bool(getattr(self.config, "null_attn", False)):
                    causal = torch.cat(
                        [torch.ones((T, 1), device=x.device, dtype=torch.bool), causal],
                        dim=1,
                    )
                scores = scores.masked_fill(~causal.view(1, 1, T, -1), ninfty)
            elif self.config.is_causal and cache is not None:
                cache_len = ksh.size(2)
                key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
                q_pos = (cache.pos - T + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                if bool(getattr(self.config, "null_attn", False)):
                    keep_null = torch.ones((1, 1, T, 1), device=x.device, dtype=torch.bool)
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
                q_chunk=int(q_chunk) if q_chunk is not None else T,
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

        B, H, T, _ = qsh.shape
        kT = ksh.size(2)
        q_chunk = max(1, int(q_chunk))

        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(T, device=qsh.device)
            k_pos_full = torch.arange(kT, device=qsh.device)
        else:
            base_q = 0
            q_pos_full = torch.arange(T, device=qsh.device)
            k_pos_full = torch.arange(kT, device=qsh.device)

        sem_scale = float(self._sem_scale) if self._sem_scale is not None else 1.0
        geo_scale = float(self._geo_scale) if self._geo_scale is not None else 1.0

        outs: list[Tensor] = []
        for i0 in range(0, T, q_chunk):
            i1 = min(T, i0 + q_chunk)
            q_pos = q_pos_full[i0:i1]

            # Key slice range when local_window is set.
            k0 = 0
            k1 = kT
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    # q_pos is a contiguous slice derived from base_q, so avoid reductions + `.item()`.
                    q_min = int(base_q + i0)
                    q_max = int(base_q + i1 - 1)
                    if self.config.is_causal:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + w)

            k_pos = k_pos_full[k0:k1]
            q_slice_sem = qsh[:, :, i0:i1, :]
            q_slice_geo = qgh[:, :, i0:i1, :]
            k_slice_sem = ksh[:, :, k0:k1, :]
            k_slice_geo = kgh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Optional memory summarization (applied consistently across K/V).
            # We summarize using the semantic K/V tensors as the reference.
            k_slice_sem, v_slice, k_pos = self._maybe_summarize_kv(
                k=k_slice_sem, v=v_slice, k_pos=k_pos
            )
            # Apply the same summarization shape to geometric keys (mean pool in blocks).
            if k_slice_geo.size(2) != k_slice_sem.size(2):
                # Fallback: recompute geometric block means to match.
                # k_pos already corresponds to the summarized sequence.
                # Keep last local_window tokens intact.
                cfg_local_window = getattr(self.config, "local_window", None)
                Tgeo = int(k_slice_geo.size(2))
                Tsem = int(k_slice_sem.size(2))
                # Ensure lw is bounded by actual tensor sizes.
                lw = int(cfg_local_window) if cfg_local_window is not None else 0
                lw = min(lw, Tgeo, Tsem)
                mem_block_val = getattr(self.config, "mem_block", None)
                if lw > 0 and lw < Tgeo and mem_block_val is not None:
                    remote_len = max(0, Tgeo - lw)
                    mb = int(mem_block_val)
                    remote = k_slice_geo[:, :, :remote_len, :]
                    local = k_slice_geo[:, :, remote_len:, :]
                    B0, H0, _Tr, D0 = remote.shape
                    n_full = remote_len // mb
                    rem = remote_len - n_full * mb
                    if n_full > 0:
                        k_full = remote[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, D0).mean(dim=3)
                    else:
                        k_full = remote.new_empty((B0, H0, 0, D0))
                    if rem > 0:
                        k_tail = remote[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
                        k_mem_geo = torch.cat([k_full, k_tail], dim=2)
                    else:
                        k_mem_geo = k_full
                    k_slice_geo = torch.cat([k_mem_geo, local], dim=2)

            null_enabled = bool(getattr(self.config, "null_attn", False))
            if null_enabled:
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=qsh.dtype, device=qsh.device)
                k_slice_sem = torch.cat([ksn, k_slice_sem], dim=2)
                k_slice_geo = torch.cat([kgn, k_slice_geo], dim=2)
                v_slice = torch.cat([vn.to(dtype=v_slice.dtype), v_slice], dim=2)

            # Fast-path: use PyTorch scaled_dot_product_attention on a composite
            # (q_cat, k_cat) representation to reduce kernel launches.
            #
            # This is particularly important on MPS where Python-driven chunking
            # can dominate runtime for moderate prompt lengths (e.g. 512+).
            #
            # Score equivalence (scale must be 1.0 to avoid SDPA's default 1/sqrt(d)):
            #   q_cat = [q_sem * sem_scale, q_geo * geo_scale]
            #   k_cat = [k_sem, k_geo]
            #   (q_cat @ k_cat^T) == (q_sem @ k_sem^T)*sem_scale + (q_geo @ k_geo^T)*geo_scale
            dropout_p = float(self.config.dropout_p) if self.training else 0.0

            if mask is None:
                attn_mask = None
                if self.config.is_causal or local_window is not None:
                    keep_tokens = torch.ones(
                        (q_pos.numel(), k_pos.numel()),
                        device=qsh.device,
                        dtype=torch.bool,
                    )
                    if self.config.is_causal:
                        keep_tokens &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                    if local_window is not None:
                        w = int(local_window)
                        if w > 0:
                            keep_tokens &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                            if not self.config.is_causal:
                                keep_tokens &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)

                    if null_enabled:
                        keep_null = torch.ones((q_pos.numel(), 1), device=qsh.device, dtype=torch.bool)
                        keep = torch.cat([keep_null, keep_tokens], dim=1)
                    else:
                        keep = keep_tokens
                    attn_mask = keep  # True = allowed (SDPA boolean semantics)

                q_cat = torch.cat(
                    [q_slice_sem * float(sem_scale), q_slice_geo * float(geo_scale)],
                    dim=-1,
                )
                k_cat = torch.cat([k_slice_sem, k_slice_geo], dim=-1)
                out = F.scaled_dot_product_attention(
                    q_cat,
                    k_cat,
                    v_slice,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
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
