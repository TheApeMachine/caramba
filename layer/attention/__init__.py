"""Attention layers

This package keeps a stable entry point (`AttentionLayer`) while allowing
multiple attention implementations to coexist, so manifests can switch attention
mechanisms without rewriting model wiring.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig, AttentionMode

if TYPE_CHECKING:
    from caramba.cache.decoupled import DecoupledLayerKVCache
    from caramba.cache.layer import LayerKVCache


class AttentionLayer(nn.Module):
    """Attention layer factory

    The constructor returns a concrete implementation based on config (standard
    vs decoupled), which keeps call sites simple while still supporting multiple
    attention research directions in one codebase.
    """

    # Attribute declarations for type checkers.
    #
    # Without these, static analyzers treat `layer.out_proj` (etc.) as coming from
    # `nn.Module.__getattr__`, which is typed as `Tensor | Module`, causing errors
    # like "Cannot access attribute in_features for class Tensor" in tests.
    #
    # IMPORTANT: these must include any attributes/methods accessed through a
    # variable typed as `AttentionLayer` (not the concrete subclasses), because
    # most callsites construct layers via `AttentionLayer(cfg)`.

    # Common attention metadata (set by AttentionBase)
    config: AttentionLayerConfig
    mode: AttentionMode
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int

    q_proj: nn.Linear | None
    k_proj: nn.Linear | None
    v_proj: nn.Linear
    out_proj: nn.Linear
    q_sem: nn.Linear | None
    k_sem: nn.Linear | None
    q_geo: nn.Linear | None
    k_geo: nn.Linear | None
    rotary: nn.Module | None
    rotary_sem: nn.Module | None
    rotary_geo: nn.Module | None

    # Decoupled-only knobs/parameters (present on all impls for typing purposes)
    decoupled_gate_logit: nn.Parameter | None
    decoupled_gate_proj: nn.Linear | None
    k_sem_null: nn.Parameter | None
    k_geo_null: nn.Parameter | None
    v_null: nn.Parameter | None

    # Internal helpers implemented by AttentionBase / concrete subclasses.
    # Stubs here exist purely so type checkers don't route through __getattr__.
    def _shape(self, x: "Tensor", head_dim: int, n_heads: int | None = None) -> "Tensor":
        raise NotImplementedError

    def _merge(self, x: "Tensor") -> "Tensor":
        raise NotImplementedError

    def _maybe_summarize_kv(
        self,
        *,
        k: "Tensor",
        v: "Tensor",
        k_pos: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor"]:
        raise NotImplementedError

    def __new__(cls, config: AttentionLayerConfig) -> "AttentionLayer":
        if cls is AttentionLayer:
            if config.mode == AttentionMode.DECOUPLED:
                from .decoupled.layer import DecoupledAttentionLayer

                impl_cls = DecoupledAttentionLayer
            else:
                from .standard.layer import StandardAttentionLayer

                impl_cls = StandardAttentionLayer
            # `nn.Module.__new__` is fine with a subclass here, but type checkers
            # can't express this factory pattern cleanly.
            return super().__new__(impl_cls)  # type: ignore[arg-type, return-value]
        return super().__new__(cls)

    # Type hint only; concrete subclasses implement the actual forward.
    def forward(  # type: ignore[override]
        self,
        x: "Tensor",
        *,
        mask: "Tensor | None" = None,
        cache: "LayerKVCache | DecoupledLayerKVCache | None" = None,
        pos_offset: int = 0,
        ctx: object | None = None,
    ) -> tuple["Tensor", "LayerKVCache | DecoupledLayerKVCache | None"]:
        raise NotImplementedError


__all__ = [
    "AttentionLayer",
    "AttentionLayerConfig",
    "AttentionMode",
]

