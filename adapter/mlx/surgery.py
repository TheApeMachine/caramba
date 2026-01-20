"""MLX Attention Surgery: Replace standard attention with fresh DBA.

This module implements the "routing hypothesis" experiment in MLX:
1. Load a pretrained Llama model's weights (FFN, embeddings, norms)
2. Initialize fresh DBA attention layers (completely random)
3. Freeze FFN/embeddings, train only attention

The hypothesis: if attention is primarily routing, the pretrained FFN layers
contain the "knowledge" and fresh attention can learn to route through them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from layer.mlx.transformer import (
    DBATransformer,
    TransformerConfig,
    create_llama_dba_config,
)


def stable_hash(s: str) -> int:
    """Deterministic hash for seeding."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


def xavier_uniform(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    """Xavier uniform initialization."""
    if len(shape) == 0:
        raise ValueError("Shape cannot be empty")
    fan_in = shape[-1] if len(shape) > 1 else shape[0]
    fan_out = shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out)) * scale

    mx.random.seed(seed)
    return mx.random.uniform(-bound, bound, shape)


class AttentionSurgeryMLX:
    """Perform attention surgery on a Llama model using MLX.

    This replaces standard attention with fresh DBA layers while preserving
    FFN weights, embeddings, and norms from the pretrained model.
    """

    def __init__(
        self,
        *,
        sem_head_dim: int = 8,
        sem_init_scale: float = 0.1,
        out_proj_scale: float = 0.02,
        decoupled_gate: bool = True,
    ):
        """Initialize surgery configuration.

        Args:
            sem_dim: Semantic projection dimension (8 dims/head for 32 heads)
            geo_dim: Geometric projection dimension (16 dims/head for 32 heads)
            v_dim: Value projection dimension
            out_proj_scale: Scale for output projection init (small to not disrupt residual)
            decoupled_gate: Enable learnable semantic/geometric gate
        """
        self.sem_head_dim = int(sem_head_dim)
        self.sem_init_scale = float(sem_init_scale)
        self.out_proj_scale = out_proj_scale
        self.decoupled_gate = decoupled_gate

    def load_llama_weights(self, weights_path: str | Path) -> dict[str, mx.array]:
        """Load Llama weights from safetensors/npz file.

        Args:
            weights_path: Path to weights file

        Returns:
            Dictionary of weight name -> mx.array
        """
        weights_path = Path(weights_path)

        if weights_path.is_dir():
            # Check for model.safetensors
            candidates = [
                weights_path / "model.safetensors",
                weights_path / "weights.npz",
            ]
            found = False
            for c in candidates:
                if c.exists():
                    weights_path = c
                    found = True
                    break
            if not found:
                raise ValueError(f"No weights found in {weights_path}")

        if weights_path.suffix == ".safetensors":
            # Use MLX's safetensors loader
            loaded = mx.load(str(weights_path))
            if isinstance(loaded, dict):
                return loaded
            elif isinstance(loaded, tuple) and len(loaded) >= 1:
                # Handle tuple return (dict, metadata)
                return loaded[0] if isinstance(loaded[0], dict) else {}
            else:
                raise ValueError(f"Unexpected return type from mx.load: {type(loaded)}")
        elif weights_path.suffix == ".npz":
            loaded = mx.load(str(weights_path))
            if isinstance(loaded, dict):
                return loaded
            elif isinstance(loaded, tuple) and len(loaded) >= 1:
                # Handle tuple return (dict, metadata)
                return loaded[0] if isinstance(loaded[0], dict) else {}
            else:
                # Convert array or other types to dict format
                raise ValueError(f"NPZ file did not return a dict: {type(loaded)}")
        else:
            raise ValueError(f"Unsupported weight format: {weights_path.suffix}")

    def create_dba_model(
        self,
        *,
        d_model: int = 2048,
        n_layers: int = 16,
        n_heads: int = 32,
        d_ff: int = 8192,
        vocab_size: int = 128256,
        rope_base: float = 500000.0,
        rope_scaling: dict[str, Any] | None = None,
        tie_embeddings: bool = False,
        geo_head_dim: int = 32,
        v_head_dim: int | None = None,
    ) -> DBATransformer:
        """Create a DBA transformer model.

        Args:
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: FFN intermediate dimension
            vocab_size: Vocabulary size
            rope_base: RoPE base frequency
            rope_scaling: Optional RoPE scaling config
            tie_embeddings: Whether to tie input/output embeddings
            geo_head_dim: Geometric head dimension (32 for fresh/A100, 64 for copy_vo)
        """
        config = create_llama_dba_config(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            sem_head_dim=self.sem_head_dim,
            v_head_dim=v_head_dim,
            geo_head_dim=geo_head_dim,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            decoupled_gate=self.decoupled_gate,
            tie_embeddings=tie_embeddings,
        )
        return DBATransformer(config)

    def apply_surgery(
        self,
        model: DBATransformer,
        teacher_weights: dict[str, mx.array],
        *,
        init_mode: str = "fresh",
    ) -> DBATransformer:
        """Apply attention surgery: copy FFN/embeddings, init fresh attention.

        Args:
            model: DBA model to populate
            teacher_weights: Pretrained Llama weights
            init_mode:
                - "fresh": Random init for all attention (pure routing hypothesis)
                - "copy_vo": Copy V/O from teacher, initialize Q/K fresh
                - "copy_vo_compress_qk": Copy V/O and derive compressed geometric Q/K from teacher (for geo_head_dim < head_dim)
                - "copy_qkvo": Copy full Q/K/V/O from teacher into geometric path (behavior-preserving; requires geo_head_dim=head_dim)

        Returns:
            Model with surgery applied
        """
        if init_mode not in ("fresh", "copy_vo", "copy_vo_compress_qk", "copy_qkvo"):
            raise NotImplementedError(f"init_mode={init_mode} not yet implemented")

        # For copy_qkvo we are strict about shapes. For copy_vo we intentionally
        # compress teacher Q/K to the configured geo_head_dim.

        # Build weight list for load_weights() - MLX requires this pattern
        # Direct attribute assignment (model.x.weight = ...) doesn't work!
        weight_list = []

        # 1. Copy token embeddings
        embed_key = self._find_key(
            teacher_weights, ["model.embed_tokens.weight", "embed_tokens.weight"]
        )
        if embed_key:
            weight_list.append(("embed_tokens.weight", teacher_weights[embed_key]))

        # 2. Copy LM head (if not tied)
        if model.lm_head is not None:
            head_key = self._find_key(
                teacher_weights, ["lm_head.weight", "model.lm_head.weight"]
            )
            if head_key:
                weight_list.append(("lm_head.weight", teacher_weights[head_key]))

        # 3. Copy final norm
        norm_key = self._find_key(teacher_weights, ["model.norm.weight", "norm.weight"])
        if norm_key:
            weight_list.append(("norm.weight", teacher_weights[norm_key]))

        # 4. Process each layer - collect weights into the list
        for i in range(len(model.layers)):
            layer_weights = self._get_layer_surgery_weights(
                model.layers[i], teacher_weights, layer_idx=i, init_mode=init_mode
            )
            weight_list.extend(layer_weights)

        # Apply all weights using tree_unflatten + update (like MLX examples)
        model.update(tree_unflatten(weight_list))

        return model

    def _find_key(
        self, weights: dict[str, mx.array], candidates: list[str]
    ) -> str | None:
        """Find first matching key from candidates."""
        for key in candidates:
            if key in weights:
                return key
        return None

    def _get_layer_surgery_weights(
        self,
        layer,  # TransformerBlock
        weights: dict[str, mx.array],
        layer_idx: int,
        init_mode: str = "fresh",
    ) -> list[tuple[str, mx.array]]:
        """Get weight tuples for a single transformer layer.

        Returns list of (name, array) tuples for use with model.load_weights().

        - Copy: input_norm, post_attn_norm, FFN weights
        - Fresh/copy init: Attention projections (based on init_mode)
        """
        weight_list = []
        prefix = f"model.layers.{layer_idx}"
        layer_prefix = f"layers.{layer_idx}"

        # Copy norms
        norm1_key = f"{prefix}.input_layernorm.weight"
        norm2_key = f"{prefix}.post_attention_layernorm.weight"

        if norm1_key in weights:
            weight_list.append((f"{layer_prefix}.norm1.weight", weights[norm1_key]))
        if norm2_key in weights:
            weight_list.append((f"{layer_prefix}.norm2.weight", weights[norm2_key]))

        # Copy FFN weights (SwiGLU)
        # Llama uses: gate_proj, up_proj, down_proj
        gate_key = f"{prefix}.mlp.gate_proj.weight"
        up_key = f"{prefix}.mlp.up_proj.weight"
        down_key = f"{prefix}.mlp.down_proj.weight"

        if gate_key in weights:
            weight_list.append((f"{layer_prefix}.ffn.w_gate.weight", weights[gate_key]))
        if up_key in weights:
            weight_list.append((f"{layer_prefix}.ffn.w_up.weight", weights[up_key]))
        if down_key in weights:
            weight_list.append((f"{layer_prefix}.ffn.w_down.weight", weights[down_key]))

        # Get attention weights based on mode
        attn = layer.attention
        attn_prefix = f"{layer_prefix}.attention"
        seed_prefix = f"layer.{layer_idx}"

        if init_mode == "copy_qkvo":
            # Behavior-preserving init: copy pretrained Q/K/V/O into geometric path,
            # initialize semantic path near-zero.
            attn_weights = self._get_attention_weights_copy_qkvo(
                attn, weights, prefix, attn_prefix, seed_prefix
            )
        elif init_mode == "copy_vo_compress_qk":
            # Compressed-geometry init: copy V/O and derive compressed Q/K.
            attn_weights = self._get_attention_weights_copy_vo_compress_qk(
                attn, weights, prefix, attn_prefix, seed_prefix
            )
        elif init_mode == "copy_vo":
            # Distillation-friendly init: copy V/O and initialize Q/K fresh.
            attn_weights = self._get_attention_weights_copy_vo(
                attn, weights, prefix, attn_prefix, seed_prefix
            )
        else:
            # Fresh init for attention (routing hypothesis)
            attn_weights = self._get_attention_weights_fresh(
                attn, attn_prefix, seed_prefix
            )

        weight_list.extend(attn_weights)
        return weight_list

    def _get_attention_weights_fresh(
        self, attn, attn_prefix: str, seed_prefix: str
    ) -> list[tuple[str, mx.array]]:
        """Get fresh attention weights (routing hypothesis).

        Returns list of (name, array) tuples for use with model.load_weights().
        Uses Xavier uniform initialization with small scale for output projection.

        Note: Geometric Q/K use compressed dimensions (geo_q_dim, geo_kv_dim).
        Values/O use DBA dimensions (v_kv_dim, v_q_dim).
        """
        weight_list = []
        d_model = attn.d_model

        # Fresh init for geometric path (compressed - matches A100 DBA)
        weight_list.append(
            (
                f"{attn_prefix}.q_geo.weight",
                xavier_uniform(
                    (attn.config.geo_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_geo"),
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_geo.weight",
                xavier_uniform(
                    (attn.config.geo_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_geo"),
                ),
            )
        )

        # V/O use DBA dimensions (paper: v_head_dim = sem+geo)
        weight_list.append(
            (
                f"{attn_prefix}.v_proj.weight",
                xavier_uniform(
                    (attn.config.v_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.v_proj"),
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.out_proj.weight",
                xavier_uniform(
                    (d_model, attn.config.v_q_dim),
                    seed=stable_hash(f"{seed_prefix}.out_proj"),
                    scale=self.out_proj_scale,
                ),
            )
        )

        # Semantic path: small random init.
        # Important: do NOT init both Q_sem and K_sem to exact zeros, or gradients are zero
        # because score_sem = Q_sem @ K_sem^T and d/dQ depends on K (and vice-versa).
        weight_list.append(
            (
                f"{attn_prefix}.q_sem.weight",
                xavier_uniform(
                    (attn.config.sem_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_sem.weight",
                xavier_uniform(
                    (attn.config.sem_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )

        # Gate is initialized in the module itself; keep it unless you want to override.

        return weight_list

    def _get_attention_weights_copy_qkvo(
        self,
        attn,
        weights: dict[str, mx.array],
        prefix: str,
        attn_prefix: str,
        seed_prefix: str,
    ) -> list[tuple[str, mx.array]]:
        """Copy full Q/K/V/O from teacher into geometric path.

        This is behavior-preserving, but requires `geo_head_dim == head_dim` (64 for Llama-3.2-1B).
        """
        weight_list = []
        d_model = attn.d_model

        # Semantic path: small random init (avoid dead start).
        weight_list.append(
            (
                f"{attn_prefix}.q_sem.weight",
                xavier_uniform(
                    (attn.config.sem_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_sem.weight",
                xavier_uniform(
                    (attn.config.sem_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )

        # Geometric Q/K/V/O from teacher
        q_key = f"{prefix}.self_attn.q_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"
        v_key = f"{prefix}.self_attn.v_proj.weight"
        o_key = f"{prefix}.self_attn.o_proj.weight"

        for kk in (q_key, k_key, v_key, o_key):
            if kk not in weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_q = weights[q_key]
        teacher_k = weights[k_key]
        teacher_v = weights[v_key]
        teacher_o = weights[o_key]

        # Check geometric Q/K shapes (using geo_q_dim/geo_kv_dim)
        if tuple(teacher_q.shape) != (attn.config.geo_q_dim, d_model):
            raise ValueError(
                f"q_proj shape mismatch: expected {(attn.config.geo_q_dim, d_model)} got {tuple(teacher_q.shape)}. "
                f"For copy_qkvo mode, set geo_head_dim=64 to match Llama."
            )
        if tuple(teacher_k.shape) != (attn.config.geo_kv_dim, d_model):
            raise ValueError(
                f"k_proj shape mismatch: expected {(attn.config.geo_kv_dim, d_model)} got {tuple(teacher_k.shape)}. "
                f"For copy_qkvo mode, set geo_head_dim=64 to match Llama."
            )
        # V/O must match the DBA v_head_dim configuration
        if tuple(teacher_v.shape) != (attn.config.v_kv_dim, d_model):
            raise ValueError(
                f"v_proj shape mismatch: expected {(attn.config.v_kv_dim, d_model)} got {tuple(teacher_v.shape)}. "
                f"For copy_* init modes, set v_head_dim=head_dim (64 for Llama-3.2-1B)."
            )
        if tuple(teacher_o.shape) != (d_model, attn.config.v_q_dim):
            raise ValueError(
                f"o_proj shape mismatch: expected {(d_model, attn.config.v_q_dim)} got {tuple(teacher_o.shape)}. "
                f"For copy_* init modes, set v_head_dim=head_dim (64 for Llama-3.2-1B)."
            )

        weight_list.append((f"{attn_prefix}.q_geo.weight", teacher_q))
        weight_list.append((f"{attn_prefix}.k_geo.weight", teacher_k))
        weight_list.append((f"{attn_prefix}.v_proj.weight", teacher_v))
        weight_list.append((f"{attn_prefix}.out_proj.weight", teacher_o))

        # Gate is initialized in the module itself; keep it unless you want to override.

        return weight_list

    def _get_attention_weights_copy_vo_compress_qk(
        self,
        attn,
        weights: dict[str, mx.array],
        prefix: str,
        attn_prefix: str,
        seed_prefix: str,
    ) -> list[tuple[str, mx.array]]:
        """Copy teacher V/O and derive compressed geometric Q/K.

        This mode is intended for the 8/32 DBA configuration:
        - `head_dim` stays full (64) for V/O (Llama-compatible)
        - `geo_head_dim` is compressed (32) for Q/K

        We initialize semantic Q/K small (to avoid a dead start), copy V/O exactly,
        and derive geometric Q/K by per-head slicing from the teacher's Q/K.
        """
        weight_list: list[tuple[str, mx.array]] = []
        d_model = attn.d_model

        # Semantic path: small random init (avoid dead start).
        weight_list.append(
            (
                f"{attn_prefix}.q_sem.weight",
                xavier_uniform(
                    (attn.config.sem_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_sem.weight",
                xavier_uniform(
                    (attn.config.sem_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )

        # Teacher keys
        q_key = f"{prefix}.self_attn.q_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"
        v_key = f"{prefix}.self_attn.v_proj.weight"
        o_key = f"{prefix}.self_attn.o_proj.weight"

        for kk in (q_key, k_key, v_key, o_key):
            if kk not in weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_q = weights[q_key]
        teacher_k = weights[k_key]
        teacher_v = weights[v_key]
        teacher_o = weights[o_key]

        head_dim = int(attn.config.computed_head_dim)
        geo_head_dim = int(attn.config.computed_geo_head_dim)
        n_heads = int(attn.config.n_heads)
        n_kv_heads = int(attn.config.computed_n_kv_heads)

        # Sanity assertions: RoPE requires even dims and geo <= teacher
        assert geo_head_dim % 2 == 0, f"geo_head_dim must be even, got {geo_head_dim}"
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        assert geo_head_dim <= head_dim, (
            f"geo_head_dim ({geo_head_dim}) cannot exceed head_dim ({head_dim})"
        )

        if geo_head_dim < head_dim:
            print(
                f"[DBA Surgery] Using sliced RoPE dims: "
                f"{geo_head_dim}/{head_dim} (teacher-consistent frequencies)"
            )

        if geo_head_dim > head_dim:
            raise ValueError(
                f"geo_head_dim ({geo_head_dim}) cannot exceed head_dim ({head_dim})"
            )

        # V/O copy exactly (requires v_head_dim=head_dim)
        if tuple(teacher_v.shape) != (attn.config.v_kv_dim, d_model):
            raise ValueError(
                f"v_proj shape mismatch: expected {(attn.config.v_kv_dim, d_model)} got {tuple(teacher_v.shape)}. "
                f"Set v_head_dim=head_dim for copy_vo_compress_qk mode."
            )
        if tuple(teacher_o.shape) != (d_model, attn.config.v_q_dim):
            raise ValueError(
                f"o_proj shape mismatch: expected {(d_model, attn.config.v_q_dim)} got {tuple(teacher_o.shape)}. "
                f"Set v_head_dim=head_dim for copy_vo_compress_qk mode."
            )

        # Geometric Q/K: per-head slice from teacher Q/K.
        # Teacher shapes: Q=(n_heads*head_dim, d_model), K=(n_kv_heads*head_dim, d_model)
        if tuple(teacher_q.shape) != (n_heads * head_dim, d_model):
            raise ValueError(
                f"q_proj shape mismatch: expected {(n_heads * head_dim, d_model)} got {tuple(teacher_q.shape)}"
            )
        if tuple(teacher_k.shape) != (n_kv_heads * head_dim, d_model):
            raise ValueError(
                f"k_proj shape mismatch: expected {(n_kv_heads * head_dim, d_model)} got {tuple(teacher_k.shape)}"
            )

        q_geo = teacher_q.reshape(n_heads, head_dim, d_model)[
            :, :geo_head_dim, :
        ].reshape(n_heads * geo_head_dim, d_model)
        k_geo = teacher_k.reshape(n_kv_heads, head_dim, d_model)[
            :, :geo_head_dim, :
        ].reshape(n_kv_heads * geo_head_dim, d_model)

        if tuple(q_geo.shape) != (attn.config.geo_q_dim, d_model):
            raise ValueError(
                f"q_geo derived shape mismatch: expected {(attn.config.geo_q_dim, d_model)} got {tuple(q_geo.shape)}"
            )
        if tuple(k_geo.shape) != (attn.config.geo_kv_dim, d_model):
            raise ValueError(
                f"k_geo derived shape mismatch: expected {(attn.config.geo_kv_dim, d_model)} got {tuple(k_geo.shape)}"
            )

        weight_list.append((f"{attn_prefix}.q_geo.weight", q_geo))
        weight_list.append((f"{attn_prefix}.k_geo.weight", k_geo))
        weight_list.append((f"{attn_prefix}.v_proj.weight", teacher_v))
        weight_list.append((f"{attn_prefix}.out_proj.weight", teacher_o))

        return weight_list

    def _get_attention_weights_copy_vo(
        self,
        attn,
        weights: dict[str, mx.array],
        prefix: str,
        attn_prefix: str,
        seed_prefix: str,
    ) -> list[tuple[str, mx.array]]:
        """Copy teacher V/O and initialize geometric Q/K fresh.

        Useful for distillation setups where Q/K should be learned from scratch
        (but we still want stable V/O and residual behavior).
        """
        weight_list: list[tuple[str, mx.array]] = []
        d_model = attn.d_model

        # Semantic path: small random init (avoid dead start).
        weight_list.append(
            (
                f"{attn_prefix}.q_sem.weight",
                xavier_uniform(
                    (attn.config.sem_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_sem.weight",
                xavier_uniform(
                    (attn.config.sem_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_sem"),
                    scale=self.sem_init_scale,
                ),
            )
        )

        # Fresh init for geometric Q/K
        weight_list.append(
            (
                f"{attn_prefix}.q_geo.weight",
                xavier_uniform(
                    (attn.config.geo_q_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.q_geo"),
                ),
            )
        )
        weight_list.append(
            (
                f"{attn_prefix}.k_geo.weight",
                xavier_uniform(
                    (attn.config.geo_kv_dim, d_model),
                    seed=stable_hash(f"{seed_prefix}.k_geo"),
                ),
            )
        )

        # Copy V/O from teacher
        v_key = f"{prefix}.self_attn.v_proj.weight"
        o_key = f"{prefix}.self_attn.o_proj.weight"
        for kk in (v_key, o_key):
            if kk not in weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_v = weights[v_key]
        teacher_o = weights[o_key]

        if tuple(teacher_v.shape) != (attn.config.v_kv_dim, d_model):
            raise ValueError(
                f"v_proj shape mismatch: expected {(attn.config.v_kv_dim, d_model)} got {tuple(teacher_v.shape)}. "
                f"Set v_head_dim=head_dim for copy_vo mode."
            )
        if tuple(teacher_o.shape) != (d_model, attn.config.v_q_dim):
            raise ValueError(
                f"o_proj shape mismatch: expected {(d_model, attn.config.v_q_dim)} got {tuple(teacher_o.shape)}. "
                f"Set v_head_dim=head_dim for copy_vo mode."
            )

        weight_list.append((f"{attn_prefix}.v_proj.weight", teacher_v))
        weight_list.append((f"{attn_prefix}.out_proj.weight", teacher_o))

        return weight_list

    def get_trainable_params(
        self, model: DBATransformer, *, include_vo: bool = True
    ) -> dict[str, mx.array]:
        """Get only the attention parameters (for gradient isolation).

        Args:
            model: The DBA transformer model
            include_vo: If True, include V/O projections. Set False for copy_vo mode
                        where V/O are copied from teacher and should stay frozen.

        Returns parameter dict containing only attention-related weights,
        which are the trainable parameters for the routing hypothesis.
        """
        trainable = {}

        for i, layer in enumerate(model.layers):
            prefix = f"layers.{i}.attention"
            attn = layer.attention

            # Train semantic path by default (retrofit)
            trainable[f"{prefix}.q_sem.weight"] = attn.q_sem.weight
            trainable[f"{prefix}.k_sem.weight"] = attn.k_sem.weight

            # Optionally train geometric Q/K as well
            trainable[f"{prefix}.q_geo.weight"] = attn.q_geo.weight
            trainable[f"{prefix}.k_geo.weight"] = attn.k_geo.weight

            # V/O only trainable if include_vo is True
            if include_vo:
                trainable[f"{prefix}.v_proj.weight"] = attn.v_proj.weight
                trainable[f"{prefix}.out_proj.weight"] = attn.out_proj.weight

            if attn.gate_logit is not None:
                trainable[f"{prefix}.gate_logit"] = attn.gate_logit

        return trainable

    def freeze_non_attention(self, model: DBATransformer) -> None:
        """Freeze all parameters except attention layers.

        In MLX, we don't explicitly freeze - we just don't include
        non-attention params in the optimizer. This method is for
        documentation/compatibility.
        """
        # MLX doesn't have requires_grad like PyTorch
        # Instead, we control what gets updated via the optimizer
        pass


def run_surgery_experiment(
    weights_path: str | Path,
    *,
    sem_head_dim: int = 8,
    geo_head_dim: int = 32,
    v_head_dim: int | None = None,
    n_layers: int = 16,
    **model_kwargs: Any,
) -> DBATransformer:
    """Convenience function to run attention surgery."""

    surgery = AttentionSurgeryMLX(sem_head_dim=sem_head_dim)

    # Load pretrained weights
    teacher_weights = surgery.load_llama_weights(weights_path)

    # Create DBA model
    model = surgery.create_dba_model(
        n_layers=n_layers,
        geo_head_dim=geo_head_dim,
        v_head_dim=v_head_dim,
        **model_kwargs,
    )

    # Apply surgery
    model = surgery.apply_surgery(model, teacher_weights, init_mode="fresh")

    return model
