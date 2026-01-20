"""Attention layers

This package keeps a stable entry point (`AttentionLayer`) while allowing
multiple attention implementations to coexist, so manifests can switch attention
mechanisms without rewriting model wiring.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from torch import nn

from config.layer import AttentionLayerConfig, AttentionMode

if TYPE_CHECKING:
    from torch import Tensor
    from cache.decoupled import DecoupledLayerKVCache
    from cache.layer import LayerKVCache


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

    qkv_proj: nn.Linear | None
    v_proj: nn.Linear | None
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

    def __new__(cls, config: AttentionLayerConfig | None = None, **kwargs: Any) -> "AttentionLayer":
        # Support both construction styles:
        #  1) AttentionLayer(cfg) where cfg is an AttentionLayerConfig
        #  2) AttentionLayer(**cfg_dict) where kwargs are the config payload (minus `type`)
        if config is None:
            if not kwargs:
                raise TypeError("AttentionLayer requires a config (AttentionLayerConfig) or keyword config fields.")
            config = AttentionLayerConfig.model_validate({"type": "AttentionLayer", **dict(kwargs)})
        elif kwargs:
            raise TypeError("AttentionLayer received both config and kwargs; provide only one.")

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

    def __init__(self, config: AttentionLayerConfig | None = None, **kwargs: Any) -> None:
        """Initialize the concrete attention layer chosen by `__new__`.

        Why this exists:
        - `AttentionLayer.__new__` returns an instance of a concrete subclass.
        - Python then calls `AttentionLayer.__init__` (not the subclass `__init__`)
          because the call site was `AttentionLayer(cfg)`.
        - Without this shim, initialization falls through to `nn.Module.__init__`,
          which doesn't accept `config` and triggers confusing downstream errors.
        """
        if config is None:
            if not kwargs:
                raise TypeError("AttentionLayer requires a config (AttentionLayerConfig) or keyword config fields.")
            config = AttentionLayerConfig.model_validate({"type": "AttentionLayer", **dict(kwargs)})
        elif kwargs:
            raise TypeError("AttentionLayer received both config and kwargs; provide only one.")

        if type(self) is AttentionLayer:
            # The factory should never be instantiated directly.
            raise TypeError("AttentionLayer is a factory and must be constructed via AttentionLayer(config).")

        # Delegate initialization to the concrete subclass returned by __new__.
        impl_init = getattr(type(self), "__init__", None)
        if impl_init is None or impl_init is AttentionLayer.__init__:
            raise TypeError(f"Concrete attention layer {type(self).__name__} does not define __init__(config).")
        impl_init(self, config)  # type: ignore[misc]

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

