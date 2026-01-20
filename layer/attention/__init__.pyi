from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from config.layer import AttentionLayerConfig, AttentionMode

class AttentionLayer(nn.Module):
    # Common attention metadata (set by AttentionBase)
    config: AttentionLayerConfig
    mode: AttentionMode
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int

    # Common module attributes exposed by all concrete implementations
    qkv_proj: nn.Linear | None
    v_proj: nn.Linear | None
    out_proj: nn.Linear

    # Decoupled-only (DBA) attributes (present as None in standard/gqa)
    q_sem: nn.Linear | None
    k_sem: nn.Linear | None
    q_geo: nn.Linear | None
    k_geo: nn.Linear | None

    decoupled_gate_logit: nn.Parameter | None
    decoupled_gate_proj: nn.Linear | None
    k_sem_null: nn.Parameter | None
    k_geo_null: nn.Parameter | None
    v_null: nn.Parameter | None

    # Rotary embeddings (present as None when disabled / unused)
    rotary: nn.Module | None
    rotary_sem: nn.Module | None
    rotary_geo: nn.Module | None

    def __init__(self, config: AttentionLayerConfig) -> None: ...

    def _shape(self, x: Tensor, head_dim: int, n_heads: int | None = ...) -> Tensor: ...
    def _merge(self, x: Tensor) -> Tensor: ...
    def _maybe_summarize_kv(
        self,
        *,
        k: Tensor,
        v: Tensor,
        k_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = ...,
        cache: Any = ...,
        pos_offset: int = ...,
        ctx: object | None = ...,
    ) -> tuple[Tensor, Any]: ...

__all__ = ["AttentionLayer", "AttentionLayerConfig", "AttentionMode"]

