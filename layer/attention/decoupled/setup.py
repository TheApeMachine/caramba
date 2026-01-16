"""DBA layer setup for the DBA (decoupled) variant

This module is purely about constructing the learnable parts of a decoupled
attention layer (projections, RoPE, gates, null tokens), keeping initialization
separate from the forward-pass logic.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from caramba.config.layer import AttentionLayerConfig
from caramba.layer.attention.base import SEM_ROPE_EVEN_DIM_ERROR
from caramba.layer.rope import RotaryEmbedding


class DecoupledSetup:
    """Decoupled layer initialization helpers

    Separating initialization into a helper class keeps the main layer focused
    on the attention math while making it easier to audit what parameters exist
    for the DBA variant.
    """

    n_heads: int
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
    _triton_warned: bool


    def init_decoupled(self, config: AttentionLayerConfig) -> None:
        """Initialize DBA projections and knobs

        DBA uses separate Q/K projections for semantic and geometric channels,
        then uses a shared V projection; that split is the core structural
        difference from standard attention.

        Precondition: `self.n_heads` must already be set, as it is used to
        determine per-head projection shapes and related knobs.
        """
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

        self.q_sem = nn.Linear(d_model, sem_dim, bias=bool(config.bias))
        if bool(getattr(config, "tie_qk", False)):
            self.k_sem = self.q_sem
        else:
            self.k_sem = nn.Linear(d_model, sem_dim, bias=bool(config.bias))

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

        if bool(getattr(config, "null_attn", False)):
            H = int(self.n_heads)
            v_head_dim = int(self._v_head_dim)
            self.k_sem_null = nn.Parameter(torch.empty((H, int(sem_head_dim))))
            self.k_geo_null = nn.Parameter(torch.empty((H, int(geo_head_dim))))
            self.v_null = nn.Parameter(torch.empty((H, int(v_head_dim))))
            nn.init.normal_(self.k_sem_null, mean=0.0, std=0.02)
            nn.init.normal_(self.k_geo_null, mean=0.0, std=0.02)
            nn.init.normal_(self.v_null, mean=0.0, std=0.02)
        else:
            self.k_sem_null = None
            self.k_geo_null = None
            self.v_null = None

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

        self._triton_warned = False

