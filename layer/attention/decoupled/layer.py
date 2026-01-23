"""Decoupled (DBA) attention layer

DBA splits attention into semantic and geometric channels and recombines them,
which is a way to test whether different “types” of relationships benefit from
different representations and inductive biases.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from console import logger

from config.layer import AttentionLayerConfig, AttentionMode
from ..base import AttentionBase
from .chunked import DecoupledSDPAChunked
from .decode import DecoupledDecode
from .memory import DecoupledMemorySummarizer
from .setup import DecoupledSetup
from .viz import DecoupledAttentionViz
from ...rope import RotaryEmbedding
from optimizer.dba_attention_triton import DecoupledAttentionTraining

if TYPE_CHECKING:
    from cache.decoupled import DecoupledLayerKVCache


_DBA_TRAINING = DecoupledAttentionTraining()


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
    _dba_train_backend: str
    _force_sdpa: bool
    _force_triton: bool
    _null_attn_enabled: bool
    _is_causal: bool
    _q_chunk_cfg: int | None
    _local_window_cfg: int | None


    def __init__(self, config: AttentionLayerConfig) -> None:
        if config.mode != AttentionMode.DECOUPLED:
            raise ValueError("DecoupledAttentionLayer requires mode=decoupled")
        super().__init__(config)
        self.init_decoupled(config)
        self._init_common_modules()
        self._viz = DecoupledAttentionViz()
        self._decode = DecoupledDecode()
        self._chunked = DecoupledSDPAChunked()
        self._init_hotpath_constants(config)

    def _init_hotpath_constants(self, config: AttentionLayerConfig) -> None:
        # Cache constant config-derived values to reduce per-forward Pydantic/getattr work.
        self._null_attn_enabled = bool(getattr(config, "null_attn", False))
        self._is_causal = bool(config.is_causal)
        self._q_chunk_cfg = getattr(config, "q_chunk", None)
        self._local_window_cfg = getattr(config, "local_window", None)

        raw_backend = getattr(config, "dba_train_backend", "auto")
        backend = str(raw_backend or "auto").lower().strip()
        allowed = {"auto", "triton", "sdpa", "metal"}
        if backend not in allowed:
            warnings.warn(
                f"Invalid dba_train_backend={raw_backend!r}; falling back to 'auto'. "
                f"Allowed values: {sorted(allowed)}",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = "auto"
        self._dba_train_backend = backend
        # "metal" is a manifest-friendly alias for the SDPA-style path on MPS.
        self._force_sdpa = (backend == "sdpa" or backend == "metal")
        self._force_triton = (backend == "triton")

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
        # Avoid `view(..., -1, ...)` here: some torch.compile/SymPy builds have hit
        # internal shape-substitution bugs (e.g. `'int' object has no attribute 'xreplace'`)
        # when -1-based inference meets symbolic shape machinery.
        #
        # Also avoid redundant device moves: parameters already live on the module device.
        H = int(self.n_heads)
        gate_bias = self.decoupled_gate_logit.reshape(1, H, 1, 1)
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1)
            gate_logit = gate_bias + dyn
        return torch.sigmoid(gate_logit.float()).to(dtype=x.dtype)

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

        # Q projections use n_heads
        q_sem = self.q_sem(x)
        qsh = self._shape(q_sem, int(sem_head_dim), n_heads=self.n_heads)

        # K projections use n_kv_heads (may be smaller for GQA)
        k_sem = self.k_sem(x)
        ksh = self._shape(k_sem, int(sem_head_dim), n_heads=self.n_kv_heads)

        if self.rotary_sem is not None:
            qsh = self.rotary_sem.rotate(qsh, pos_offset)
            ksh = self.rotary_sem.rotate(ksh, pos_offset)

        q_geo = self.q_geo(x)
        qgh = self._shape(q_geo, int(geo_head_dim), n_heads=self.n_heads)

        k_geo = self.k_geo(x)
        kgh = self._shape(k_geo, int(geo_head_dim), n_heads=self.n_kv_heads)

        if self.rotary_geo is not None:
            qgh = self.rotary_geo.rotate(qgh, pos_offset)
            kgh = self.rotary_geo.rotate(kgh, pos_offset)

        # V projection uses n_kv_heads
        v = self.v_proj(x)
        vh = self._shape(v, int(v_head_dim), n_heads=self.n_kv_heads)

        qsh = self._apply_logit_scale(qsh)
        qgh = self._apply_logit_scale(qgh)

        # Fold all query-side scaling into q_sem/q_geo once (better fusion + fewer temporaries):
        # - base sem/geo scales (part of the DBA math)
        # - optional learned gate (extra architecture knob)
        #
        # Doing this once keeps downstream backends (decode/triton/sdpa) simpler and reduces
        # extra elementwise work in the hot path.
        sem_scale = float(self._sem_scale)
        geo_scale = float(self._geo_scale)
        g = self._decoupled_gate(x)
        if g is not None:
            g2 = g * 2.0
            qsh = qsh * (g2 * sem_scale)
            qgh = qgh * ((2.0 - g2) * geo_scale)
        else:
            qsh = qsh * sem_scale
            qgh = qgh * geo_scale
        # After folding scales into Q, downstream uses unit scales.
        sem_scale = 1.0
        geo_scale = 1.0

        cache_pos = None
        if cache is not None:
            cache_pos = int(cache.pos)
            old_len = int(cache.pos)
            # Store K/V in cache BEFORE GQA expansion to save memory
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
                # Note: fused decode kernel handles GQA expansion internally
                out_fused = self._decode.run(
                    q_sem=qsh,
                    q_geo=qgh,
                    cache=cache,
                    n_heads=int(self.n_heads),
                    sem_head_dim=int(sem_head_dim),
                    geo_head_dim=int(geo_head_dim),
                    v_head_dim=int(v_head_dim),
                        sem_scale=float(sem_scale),
                        geo_scale=float(geo_scale),
                    decode_block=int(decode_block),
                    null_enabled=bool(null_enabled),
                    null_kv=self._null_kv_tensors,
                )
                y = self.out_proj(self._merge(out_fused))
                self._viz.record_activation_sample(ctx=ctx, layer=self, y=y)
                return y, cache
            if old_len > 0:
                # Retrieve from cache and reshape with n_kv_heads
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, int(sem_head_dim), n_heads=self.n_kv_heads)
                kgh = self._shape(k_geo_all, int(geo_head_dim), n_heads=self.n_kv_heads)
                vh = self._shape(v_all, int(v_head_dim), n_heads=self.n_kv_heads)

        # Expand K and V for GQA: repeat each KV head to match Q heads
        # This happens AFTER cache handling to avoid storing expanded tensors
        if self.group_size > 1:
            ksh = ksh.repeat_interleave(self.group_size, dim=1)
            kgh = kgh.repeat_interleave(self.group_size, dim=1)
            vh = vh.repeat_interleave(self.group_size, dim=1)

        q_chunk = q_chunk_override if q_chunk_override is not None else self._q_chunk_cfg
        local_window = local_window_override if local_window_override is not None else self._local_window_cfg

        dropout_p = float(self.config.dropout_p) if self.training else 0.0
        null_enabled = bool(self._null_attn_enabled)

        # `dba_train_backend` controls which DBA training backend to use:
        # - "auto": prefer the Triton training kernel when eligible (default)
        # - "triton": prefer the Triton training kernel (warn/fallback if ineligible)
        # - "sdpa": force scaled dot-product attention
        dba_backend = self._dba_train_backend
        force_sdpa = bool(self._force_sdpa)
        force_triton = bool(self._force_triton)
        is_triton_eligible = (
            q_chunk is None
            and local_window is None
            and mask is None
            and cache is None
            and x.device.type == "cuda"
            and self.training
            and not null_enabled
        )

        if force_triton and not is_triton_eligible and not self._triton_warned:
            self._triton_warned = True
            warnings.warn(
                "dba_train_backend='triton' requested, but Triton training kernel is not eligible "
                "for this call; falling back to SDPA. Eligibility requires: training=True, "
                "device=cuda, no q_chunk/local_window/mask/cache, and null_attn disabled.",
                RuntimeWarning,
                stacklevel=2,
            )

        if (not force_sdpa) and is_triton_eligible:
            # Select DBA training backend. Default to Triton for deterministic behavior.
            # - "triton": Triton FlashAttention kernel (default, deterministic)
            # - "sdpa": PyTorch scaled_dot_product_attention
            # - "auto": same as "triton" (no auto-benchmarking to ensure reproducibility)
            if dba_backend == "sdpa":
                backend = "sdpa"
            else:
                # "triton" or "auto" -> use Triton for deterministic, reproducible results
                backend = "triton"

            if backend == "sdpa":
                q_cat = torch.cat([qsh, qgh], dim=-1)
                k_cat = torch.cat([ksh, kgh], dim=-1)
                # Some device backends (notably MPS/Metal) are sensitive to Q/K/V head-dim
                # mismatches. DBA often uses a smaller Q/K dim (sem+geo) than V dim (attn_dim),
                # so we pad Q/K up to V's head dim with zeros. This preserves the attention math
                # (extra dims contribute nothing) while keeping kernels shape-safe.
                qk_dim = int(q_cat.size(-1))
                if qk_dim != int(v_head_dim):
                    if qk_dim > int(v_head_dim):
                        raise RuntimeError(
                            f"DBA Q/K head dim ({qk_dim}) must be <= V head dim ({int(v_head_dim)}). "
                            "Decrease sem_dim+geo_dim or increase attn_dim."
                        )
                    pad = int(v_head_dim) - qk_dim
                    q_cat = F.pad(q_cat, (0, pad))
                    k_cat = F.pad(k_cat, (0, pad))
                out = F.scaled_dot_product_attention(
                    q_cat,
                    k_cat,
                    vh,
                    attn_mask=None,
                    dropout_p=float(dropout_p),
                    is_causal=bool(self._is_causal) and int(T) > 1,
                    scale=1.0,
                )
                if ctx is not None:
                    self._viz.record_attention_matrix(ctx=ctx, layer=self, q_cat=q_cat, k_cat=k_cat)
            else:
                out = _DBA_TRAINING.run(
                    q_sem=qsh,
                    q_geo=qgh,
                    k_sem=ksh,
                    k_geo=kgh,
                    v=vh,
                    causal=bool(self._is_causal) and int(T) > 1,
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                    dropout_p=float(dropout_p),
                )
        elif q_chunk is None and local_window is None:
            q_cat = torch.cat([qsh, qgh], dim=-1)
            k_cat = torch.cat([ksh, kgh], dim=-1)
            if null_enabled:
                ksn, kgn, vn = self._null_kv_tensors(B=B, dtype=x.dtype, device=x.device)
                k_null = torch.cat([ksn, kgn], dim=-1)
                k_cat = torch.cat([k_null.to(dtype=k_cat.dtype), k_cat], dim=2)
                vh = torch.cat([vn.to(dtype=vh.dtype), vh], dim=2)
                if mask is not None:
                    keep_null = torch.ones((*mask.shape[:-1], 1), device=mask.device, dtype=torch.bool)
                    mask = torch.cat([keep_null, mask], dim=-1)

            # Some SDPA backends (notably MPS/Metal) are sensitive to Q/K/V head-dim
            # mismatches. DBA often uses a smaller Q/K dim (sem+geo) than V dim (attn_dim),
            # so we pad Q/K up to V's head dim with zeros. This preserves the attention math
            # (extra dims contribute nothing) while keeping kernels shape-safe.
            qk_dim = int(q_cat.size(-1))
            if qk_dim != int(v_head_dim):
                if qk_dim > int(v_head_dim):
                    raise RuntimeError(
                        f"DBA Q/K head dim ({qk_dim}) must be <= V head dim ({int(v_head_dim)}). "
                        "Decrease sem_dim+geo_dim or increase attn_dim."
                    )
                pad = int(v_head_dim) - qk_dim
                q_cat = F.pad(q_cat, (0, pad))
                k_cat = F.pad(k_cat, (0, pad))

            can_use_is_causal = (mask is None) and (cache is None) and (not null_enabled) and bool(self._is_causal) and int(T) > 1 and x.device.type == "cuda"
            attn_mask = mask
            is_causal = bool(can_use_is_causal)
            if attn_mask is None and (not is_causal) and bool(self._is_causal) and int(T) > 1:
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
            if ctx is not None:
                self._viz.record_attention_matrix(ctx=ctx, layer=self, q_cat=q_cat, k_cat=k_cat)
        else:
            out = self._chunked.run(
                qsh=qsh,
                ksh=ksh,
                qgh=qgh,
                kgh=kgh,
                vh=vh,
                is_causal=bool(self._is_causal),
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
        if ctx is not None:
            self._viz.record_activation_sample(ctx=ctx, layer=self, y=y)
        return y, cache

