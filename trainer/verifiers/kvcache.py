"""KV-cache footprint estimation helpers for verification."""

from __future__ import annotations

from torch import nn

from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig
from caramba.config.layer import AttentionMode
from carmath import bytes_per_kind
from caramba.layer.attention import AttentionLayer


def estimate_model_kvcache_bytes(
    model: nn.Module, policy: KVCachePolicyConfig, n_layers: int
) -> int:
    """Estimate bytes per token for a standard model's KV cache."""
    k_dim = 0
    v_dim = 0
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            k_dim = module.config.kv_heads * module.config.head_dim
            v_dim = k_dim
            break

    k_bytes = k_dim * bytes_per_kind(policy.k.kind.value)
    v_bytes = v_dim * bytes_per_kind(policy.v.kind.value)
    return int((k_bytes + v_bytes) * int(n_layers))


def estimate_model_kvcache_bytes_decoupled(
    model: nn.Module, policy: KVCachePolicyDecoupledConfig, n_layers: int
) -> int:
    """Estimate bytes per token for a DBA model's KV cache."""
    sem_dim = 0
    geo_dim = 0
    v_dim = 0
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            cfg = module.config
            if cfg.mode == AttentionMode.DECOUPLED:
                sem_dim = cfg.sem_dim or cfg.d_model
                geo_dim = cfg.geo_dim or cfg.d_model
                v_dim = cfg.v_dim
            else:
                sem_dim = cfg.kv_heads * cfg.head_dim
                geo_dim = 0
                v_dim = sem_dim
            break

    k_sem_bytes = sem_dim * bytes_per_kind(policy.k_sem.kind.value)
    k_geo_bytes = geo_dim * bytes_per_kind(policy.k_geo.kind.value)
    v_bytes = v_dim * bytes_per_kind(policy.v.kind.value)
    return int((k_sem_bytes + k_geo_bytes + v_bytes) * int(n_layers))

