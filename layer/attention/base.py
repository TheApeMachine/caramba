"""Shared implementation pieces for attention layers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.carmath import neg_inf
from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.console import logger
from caramba.layer.attention import AttentionLayer
from caramba.layer.rope import RotaryEmbedding

if TYPE_CHECKING:
    from cache.decoupled import DecoupledLayerKVCache
    from cache.layer import LayerKVCache


# Error message constants (keep exact wording for tests/log searchability).
SEM_ROPE_EVEN_DIM_ERROR = "Decoupled mode with semantic RoPE requires even sem_head_dim"

# Lazy-cached reference to avoid per-call import overhead
_InferContext: type | None = None


def _get_infer_context_type() -> type:
    """Get the InferContext type, caching it on first access."""
    global _InferContext
    if _InferContext is None:
        # Import from the `caramba` package namespace. Importing `infer.context`
        # would create/resolve a different module, causing `isinstance(ctx, InferContext)`
        # checks to fail and KV caches to never be consumed.
        from caramba.infer.context import InferContext

        _InferContext = InferContext
    return _InferContext


class AttentionBase(AttentionLayer):
    """Common init + helpers for all attention implementations."""

    # Shared attributes used across implementations
    config: AttentionLayerConfig
    mode: AttentionMode
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int
    dropout: nn.Dropout

    # Optional long-sequence memory summarization modules.
    mem_k_proj: nn.Module | None
    mem_v_proj: nn.Module | None

    # Optional learned temperature scaling per head
    logit_scale: nn.Parameter | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        # Initialize the actual nn.Module base. We avoid relying on the factory
        # `AttentionLayer` having any particular `__init__` signature.
        nn.Module.__init__(self)
        self.config = config
        self.mode = config.mode
        self.n_heads = int(config.n_heads)
        self.n_kv_heads = int(config.kv_heads)
        self.head_dim = int(config.head_dim)
        self.dropout = nn.Dropout(float(config.dropout_p))
        self.mem_k_proj = None
        self.mem_v_proj = None

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        self.group_size = self.n_heads // self.n_kv_heads

        # Subclasses must initialize projections/scales.
        self.logit_scale = None

    def _init_common_modules(self) -> None:
        """Initialize modules shared across modes (after projections exist)."""
        if bool(getattr(self.config, "learned_temp", False)):
            self.logit_scale = nn.Parameter(torch.zeros(self.n_heads))
        self._init_memory_summarizer()

    def _init_memory_summarizer(self) -> None:
        """Initialize optional modules for mem_block summarization."""
        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "linear":
            d = int(self.head_dim)
            self.mem_k_proj = nn.Linear(d, d, bias=False)
            self.mem_v_proj = nn.Linear(d, d, bias=False)
            # Initialize as identity so "linear" starts equivalent to mean pooling.
            nn.init.eye_(cast(nn.Linear, self.mem_k_proj).weight)
            nn.init.eye_(cast(nn.Linear, self.mem_v_proj).weight)
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
        """Summarize older KV blocks into memory tokens for long sequences."""
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

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        cache: "LayerKVCache | DecoupledLayerKVCache | None" = None,
        pos_offset: int = 0,
        ctx: object | None = None,
    ) -> tuple[Tensor, "LayerKVCache | DecoupledLayerKVCache | None"]:
        """Compute attention and return output with updated cache."""
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
            out, updated = self._forward_decoupled(  # type: ignore[attr-defined]
                x,
                mask=mask,
                cache=decoupled_cache,
                pos_offset=pos_offset,
                ctx=ctx,
                q_chunk_override=q_chunk_override,
                local_window_override=local_window_override,
                decode_block_override=decode_block_override,
            )
            return out, updated

        standard_cache = cast("LayerKVCache | None", cache)
        out, updated = self._forward_standard(  # type: ignore[attr-defined]
            x,
            mask=mask,
            cache=standard_cache,
            pos_offset=pos_offset,
            ctx=ctx,
            q_chunk_override=q_chunk_override,
            local_window_override=local_window_override,
        )
        return out, updated

