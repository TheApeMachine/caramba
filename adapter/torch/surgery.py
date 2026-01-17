"""PyTorch Attention Surgery: Replace standard attention with fresh DBA.

This module implements the "routing hypothesis" experiment in PyTorch:
1. Load a pretrained Llama model's weights (FFN, embeddings, norms)
2. Initialize fresh DBA attention layers (completely random or from teacher)
3. Freeze FFN/embeddings, train only attention

The hypothesis: if attention is primarily routing, the pretrained FFN layers
contain the "knowledge" and fresh attention can learn to route through them.

This is a PyTorch port of the MLX implementation in adapter/mlx/surgery.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# For loading safetensors
try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def stable_hash(s: str) -> int:
    """Deterministic hash for seeding (matches MLX version)."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


def xavier_uniform_(
    tensor: torch.Tensor,
    seed: int,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Xavier uniform initialization with explicit seed.

    Args:
        tensor: Tensor to initialize in-place
        seed: Random seed for reproducibility
        scale: Scale factor for the initialization bounds
        generator: Optional PyTorch generator (created from seed if None)

    Returns:
        The initialized tensor
    """
    fan_in = tensor.shape[-1] if tensor.ndim > 1 else tensor.shape[0]
    fan_out = tensor.shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out)) * scale

    if generator is None:
        generator = torch.Generator()
    generator.manual_seed(seed)

    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)
    return tensor


def xavier_uniform_tensor(
    shape: tuple[int, ...],
    seed: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a new tensor with Xavier uniform initialization.

    Args:
        shape: Shape of the tensor to create
        seed: Random seed for reproducibility
        scale: Scale factor for the initialization bounds
        dtype: Data type of the tensor
        device: Device to create the tensor on

    Returns:
        New tensor with Xavier uniform initialization
    """
    tensor = torch.empty(shape, dtype=dtype, device=device)
    return xavier_uniform_(tensor, seed, scale)


class SurgeryConfig:
    """Configuration for attention surgery."""

    def __init__(
        self,
        *,
        # Model architecture
        d_model: int = 2048,
        n_layers: int = 16,
        n_heads: int = 32,
        n_kv_heads: int | None = None,  # defaults to n_heads (no GQA)
        d_ff: int = 8192,
        vocab_size: int = 128256,

        # Head dimensions
        head_dim: int | None = None,  # Teacher head dim, defaults to d_model // n_heads
        sem_head_dim: int = 8,        # Semantic compression per head
        geo_head_dim: int = 32,       # Geometric compression per head
        v_head_dim: int | None = None,  # Value dim, defaults to sem + geo

        # RoPE
        rope_base: float = 500000.0,
        rope_scaling: dict[str, Any] | None = None,

        # DBA options
        decoupled_gate: bool = True,
        tie_embeddings: bool = False,

        # Init scales
        sem_init_scale: float = 0.1,
        out_proj_scale: float = 0.02,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size

        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.sem_head_dim = sem_head_dim
        self.geo_head_dim = geo_head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else sem_head_dim + geo_head_dim

        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.decoupled_gate = decoupled_gate
        self.tie_embeddings = tie_embeddings

        self.sem_init_scale = sem_init_scale
        self.out_proj_scale = out_proj_scale

    @property
    def sem_q_dim(self) -> int:
        """Total semantic Q dimension."""
        return self.n_heads * self.sem_head_dim

    @property
    def sem_kv_dim(self) -> int:
        """Total semantic KV dimension (uses n_kv_heads for GQA)."""
        return self.n_kv_heads * self.sem_head_dim

    @property
    def geo_q_dim(self) -> int:
        """Total geometric Q dimension."""
        return self.n_heads * self.geo_head_dim

    @property
    def geo_kv_dim(self) -> int:
        """Total geometric KV dimension (uses n_kv_heads for GQA)."""
        return self.n_kv_heads * self.geo_head_dim

    @property
    def v_q_dim(self) -> int:
        """Total value Q dimension (output projection input)."""
        return self.n_heads * self.v_head_dim

    @property
    def v_kv_dim(self) -> int:
        """Total value KV dimension."""
        return self.n_kv_heads * self.v_head_dim


class AttentionSurgeryTorch:
    """Perform attention surgery on a Llama model using PyTorch.

    This replaces standard attention with fresh DBA layers while preserving
    FFN weights, embeddings, and norms from the pretrained model.
    """

    def __init__(self, config: SurgeryConfig):
        """Initialize surgery with configuration.

        Args:
            config: Surgery configuration specifying dimensions and init parameters
        """
        self.config = config

    def load_teacher_weights(
        self,
        weights_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load teacher weights from safetensors/pt file.

        Args:
            weights_path: Path to weights file or directory
            device: Device to load weights onto

        Returns:
            Dictionary of weight name -> tensor
        """
        weights_path = Path(weights_path)
        device = torch.device(device) if isinstance(device, str) else device

        if weights_path.is_dir():
            # Check for common weight file names
            candidates = [
                weights_path / "model.safetensors",
                weights_path / "pytorch_model.bin",
                weights_path / "model.pt",
                weights_path / "weights.pt",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                raise ValueError(f"No weights found in {weights_path}")
            weights_path = found

        if weights_path.suffix == ".safetensors":
            if not HAS_SAFETENSORS:
                raise ImportError("safetensors not installed. Run: pip install safetensors")
            weights = load_safetensors(str(weights_path), device=str(device))
        elif weights_path.suffix in (".pt", ".bin"):
            weights = torch.load(str(weights_path), map_location=device, weights_only=False)
            # Handle nested state_dict format
            if "model" in weights:
                weights = weights["model"]
            elif "state_dict" in weights:
                weights = weights["state_dict"]
        else:
            raise ValueError(f"Unsupported weight format: {weights_path.suffix}")

        return weights

    def apply_surgery(
        self,
        student_state_dict: dict[str, torch.Tensor],
        teacher_weights: dict[str, torch.Tensor],
        *,
        init_mode: str = "fresh",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Apply attention surgery: copy FFN/embeddings, init fresh/copied attention.

        Args:
            student_state_dict: State dict of the DBA student model to populate
            teacher_weights: Pretrained teacher (Llama) weights
            init_mode:
                - "fresh": Random init for all attention (pure routing hypothesis)
                - "copy_vo": Copy V/O from teacher, initialize Q/K fresh
                - "copy_vo_compress_qk": Copy V/O and derive compressed geometric Q/K
                - "copy_qkvo": Copy full Q/K/V/O from teacher into geometric path
            device: Device for new tensors
            dtype: Data type for new tensors

        Returns:
            Updated state dict with surgery applied
        """
        if init_mode not in ("fresh", "copy_vo", "copy_vo_compress_qk", "copy_qkvo"):
            raise ValueError(f"Unknown init_mode: {init_mode}")

        device = torch.device(device) if isinstance(device, str) else device
        result = dict(student_state_dict)  # Copy to avoid mutation

        # 1. Copy token embeddings
        embed_key = self._find_key(
            teacher_weights, ["model.embed_tokens.weight", "embed_tokens.weight"]
        )
        if embed_key:
            result["embed_tokens.weight"] = teacher_weights[embed_key].to(device=device, dtype=dtype)

        # 2. Copy LM head (if not tied)
        if not self.config.tie_embeddings:
            head_key = self._find_key(
                teacher_weights, ["lm_head.weight", "model.lm_head.weight"]
            )
            if head_key:
                result["lm_head.weight"] = teacher_weights[head_key].to(device=device, dtype=dtype)

        # 3. Copy final norm
        norm_key = self._find_key(teacher_weights, ["model.norm.weight", "norm.weight"])
        if norm_key:
            result["norm.weight"] = teacher_weights[norm_key].to(device=device, dtype=dtype)

        # 4. Process each layer
        for layer_idx in range(self.config.n_layers):
            layer_weights = self._get_layer_surgery_weights(
                teacher_weights,
                layer_idx=layer_idx,
                init_mode=init_mode,
                device=device,
                dtype=dtype,
            )
            result.update(layer_weights)

        return result

    def _find_key(
        self, weights: dict[str, torch.Tensor], candidates: list[str]
    ) -> str | None:
        """Find first matching key from candidates."""
        for key in candidates:
            if key in weights:
                return key
        return None

    def _get_layer_surgery_weights(
        self,
        teacher_weights: dict[str, torch.Tensor],
        layer_idx: int,
        init_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Get weight dict for a single transformer layer.

        - Copy: input_norm, post_attn_norm, FFN weights
        - Fresh/copy init: Attention projections (based on init_mode)
        """
        result = {}
        teacher_prefix = f"model.layers.{layer_idx}"
        student_prefix = f"layers.{layer_idx}"
        seed_prefix = f"layer.{layer_idx}"

        # Copy norms
        norm1_key = f"{teacher_prefix}.input_layernorm.weight"
        norm2_key = f"{teacher_prefix}.post_attention_layernorm.weight"

        if norm1_key in teacher_weights:
            result[f"{student_prefix}.norm1.weight"] = teacher_weights[norm1_key].to(device=device, dtype=dtype)
        if norm2_key in teacher_weights:
            result[f"{student_prefix}.norm2.weight"] = teacher_weights[norm2_key].to(device=device, dtype=dtype)

        # Copy FFN weights (SwiGLU: gate_proj, up_proj, down_proj)
        gate_key = f"{teacher_prefix}.mlp.gate_proj.weight"
        up_key = f"{teacher_prefix}.mlp.up_proj.weight"
        down_key = f"{teacher_prefix}.mlp.down_proj.weight"

        if gate_key in teacher_weights:
            result[f"{student_prefix}.ffn.w_gate.weight"] = teacher_weights[gate_key].to(device=device, dtype=dtype)
        if up_key in teacher_weights:
            result[f"{student_prefix}.ffn.w_up.weight"] = teacher_weights[up_key].to(device=device, dtype=dtype)
        if down_key in teacher_weights:
            result[f"{student_prefix}.ffn.w_down.weight"] = teacher_weights[down_key].to(device=device, dtype=dtype)

        # Get attention weights based on mode
        attn_prefix = f"{student_prefix}.attention"

        if init_mode == "copy_qkvo":
            attn_weights = self._get_attention_weights_copy_qkvo(
                teacher_weights, teacher_prefix, attn_prefix, seed_prefix, device, dtype
            )
        elif init_mode == "copy_vo_compress_qk":
            attn_weights = self._get_attention_weights_copy_vo_compress_qk(
                teacher_weights, teacher_prefix, attn_prefix, seed_prefix, device, dtype
            )
        elif init_mode == "copy_vo":
            attn_weights = self._get_attention_weights_copy_vo(
                teacher_weights, teacher_prefix, attn_prefix, seed_prefix, device, dtype
            )
        else:  # fresh
            attn_weights = self._get_attention_weights_fresh(
                attn_prefix, seed_prefix, device, dtype
            )

        result.update(attn_weights)
        return result

    def _get_attention_weights_fresh(
        self,
        attn_prefix: str,
        seed_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Get fresh attention weights (routing hypothesis).

        Uses Xavier uniform initialization with small scale for output projection.
        """
        cfg = self.config
        result = {}

        # Geometric Q/K (compressed)
        result[f"{attn_prefix}.q_geo.weight"] = xavier_uniform_tensor(
            (cfg.geo_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_geo"),
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_geo.weight"] = xavier_uniform_tensor(
            (cfg.geo_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_geo"),
            dtype=dtype,
            device=device,
        )

        # V/O projections
        result[f"{attn_prefix}.v_proj.weight"] = xavier_uniform_tensor(
            (cfg.v_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.v_proj"),
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.out_proj.weight"] = xavier_uniform_tensor(
            (cfg.d_model, cfg.v_q_dim),
            seed=stable_hash(f"{seed_prefix}.out_proj"),
            scale=cfg.out_proj_scale,
            dtype=dtype,
            device=device,
        )

        # Semantic Q/K (small scale to avoid dead start but not dominate initially)
        result[f"{attn_prefix}.q_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )

        # Gate (if enabled) - initialized to 0.5 via sigmoid(0)
        if cfg.decoupled_gate:
            result[f"{attn_prefix}.gate_logit"] = torch.zeros(
                cfg.n_heads, dtype=dtype, device=device
            )

        return result

    def _get_attention_weights_copy_qkvo(
        self,
        teacher_weights: dict[str, torch.Tensor],
        teacher_prefix: str,
        attn_prefix: str,
        seed_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Copy full Q/K/V/O from teacher into geometric path.

        This is behavior-preserving, but requires geo_head_dim == head_dim.
        """
        cfg = self.config
        result = {}

        # Semantic path: small random init
        result[f"{attn_prefix}.q_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )

        # Get teacher Q/K/V/O
        q_key = f"{teacher_prefix}.self_attn.q_proj.weight"
        k_key = f"{teacher_prefix}.self_attn.k_proj.weight"
        v_key = f"{teacher_prefix}.self_attn.v_proj.weight"
        o_key = f"{teacher_prefix}.self_attn.o_proj.weight"

        for kk in (q_key, k_key, v_key, o_key):
            if kk not in teacher_weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_q = teacher_weights[q_key]
        teacher_k = teacher_weights[k_key]
        teacher_v = teacher_weights[v_key]
        teacher_o = teacher_weights[o_key]

        # Validate shapes
        expected_q_shape = (cfg.geo_q_dim, cfg.d_model)
        expected_k_shape = (cfg.geo_kv_dim, cfg.d_model)
        expected_v_shape = (cfg.v_kv_dim, cfg.d_model)
        expected_o_shape = (cfg.d_model, cfg.v_q_dim)

        if tuple(teacher_q.shape) != expected_q_shape:
            raise ValueError(
                f"q_proj shape mismatch: expected {expected_q_shape} got {tuple(teacher_q.shape)}. "
                f"For copy_qkvo mode, set geo_head_dim={cfg.head_dim} to match teacher."
            )
        if tuple(teacher_k.shape) != expected_k_shape:
            raise ValueError(
                f"k_proj shape mismatch: expected {expected_k_shape} got {tuple(teacher_k.shape)}. "
                f"For copy_qkvo mode, set geo_head_dim={cfg.head_dim} to match teacher."
            )
        if tuple(teacher_v.shape) != expected_v_shape:
            raise ValueError(
                f"v_proj shape mismatch: expected {expected_v_shape} got {tuple(teacher_v.shape)}. "
                f"For copy_* modes, set v_head_dim={cfg.head_dim} to match teacher."
            )
        if tuple(teacher_o.shape) != expected_o_shape:
            raise ValueError(
                f"o_proj shape mismatch: expected {expected_o_shape} got {tuple(teacher_o.shape)}. "
                f"For copy_* modes, set v_head_dim={cfg.head_dim} to match teacher."
            )

        result[f"{attn_prefix}.q_geo.weight"] = teacher_q.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.k_geo.weight"] = teacher_k.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.v_proj.weight"] = teacher_v.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.out_proj.weight"] = teacher_o.to(device=device, dtype=dtype)

        # Gate (if enabled)
        if cfg.decoupled_gate:
            result[f"{attn_prefix}.gate_logit"] = torch.zeros(
                cfg.n_heads, dtype=dtype, device=device
            )

        return result

    def _get_attention_weights_copy_vo_compress_qk(
        self,
        teacher_weights: dict[str, torch.Tensor],
        teacher_prefix: str,
        attn_prefix: str,
        seed_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Copy teacher V/O and derive compressed geometric Q/K.

        This mode is intended for configs like sem=8, geo=32:
        - head_dim stays full (64) for V/O (Llama-compatible)
        - geo_head_dim is compressed (32) for Q/K

        We derive geometric Q/K by per-head slicing from teacher's Q/K.
        """
        cfg = self.config
        result = {}

        # Semantic path: small random init
        result[f"{attn_prefix}.q_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )

        # Get teacher Q/K/V/O
        q_key = f"{teacher_prefix}.self_attn.q_proj.weight"
        k_key = f"{teacher_prefix}.self_attn.k_proj.weight"
        v_key = f"{teacher_prefix}.self_attn.v_proj.weight"
        o_key = f"{teacher_prefix}.self_attn.o_proj.weight"

        for kk in (q_key, k_key, v_key, o_key):
            if kk not in teacher_weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_q = teacher_weights[q_key]
        teacher_k = teacher_weights[k_key]
        teacher_v = teacher_weights[v_key]
        teacher_o = teacher_weights[o_key]

        head_dim = cfg.head_dim
        geo_head_dim = cfg.geo_head_dim
        n_heads = cfg.n_heads
        n_kv_heads = cfg.n_kv_heads

        # Sanity checks
        if geo_head_dim % 2 != 0:
            raise ValueError(f"geo_head_dim must be even for RoPE, got {geo_head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if geo_head_dim > head_dim:
            raise ValueError(f"geo_head_dim ({geo_head_dim}) cannot exceed head_dim ({head_dim})")

        if geo_head_dim < head_dim:
            print(
                f"[DBA Surgery] Using sliced RoPE dims: "
                f"{geo_head_dim}/{head_dim} (teacher-consistent frequencies)"
            )

        # Validate V/O shapes (must match v_head_dim = head_dim for this mode)
        expected_v_shape = (cfg.v_kv_dim, cfg.d_model)
        expected_o_shape = (cfg.d_model, cfg.v_q_dim)

        if tuple(teacher_v.shape) != expected_v_shape:
            raise ValueError(
                f"v_proj shape mismatch: expected {expected_v_shape} got {tuple(teacher_v.shape)}. "
                f"Set v_head_dim={head_dim} for copy_vo_compress_qk mode."
            )
        if tuple(teacher_o.shape) != expected_o_shape:
            raise ValueError(
                f"o_proj shape mismatch: expected {expected_o_shape} got {tuple(teacher_o.shape)}. "
                f"Set v_head_dim={head_dim} for copy_vo_compress_qk mode."
            )

        # Validate teacher Q/K shapes
        expected_teacher_q = (n_heads * head_dim, cfg.d_model)
        expected_teacher_k = (n_kv_heads * head_dim, cfg.d_model)

        if tuple(teacher_q.shape) != expected_teacher_q:
            raise ValueError(
                f"q_proj shape mismatch: expected {expected_teacher_q} got {tuple(teacher_q.shape)}"
            )
        if tuple(teacher_k.shape) != expected_teacher_k:
            raise ValueError(
                f"k_proj shape mismatch: expected {expected_teacher_k} got {tuple(teacher_k.shape)}"
            )

        # Derive compressed geometric Q/K by per-head slicing
        # Teacher Q shape: (n_heads * head_dim, d_model)
        # Reshape to (n_heads, head_dim, d_model), slice [:, :geo_head_dim, :], reshape back
        q_geo = teacher_q.reshape(n_heads, head_dim, cfg.d_model)[
            :, :geo_head_dim, :
        ].reshape(n_heads * geo_head_dim, cfg.d_model)

        k_geo = teacher_k.reshape(n_kv_heads, head_dim, cfg.d_model)[
            :, :geo_head_dim, :
        ].reshape(n_kv_heads * geo_head_dim, cfg.d_model)

        # Validate derived shapes
        if tuple(q_geo.shape) != (cfg.geo_q_dim, cfg.d_model):
            raise ValueError(
                f"q_geo derived shape mismatch: expected {(cfg.geo_q_dim, cfg.d_model)} got {tuple(q_geo.shape)}"
            )
        if tuple(k_geo.shape) != (cfg.geo_kv_dim, cfg.d_model):
            raise ValueError(
                f"k_geo derived shape mismatch: expected {(cfg.geo_kv_dim, cfg.d_model)} got {tuple(k_geo.shape)}"
            )

        result[f"{attn_prefix}.q_geo.weight"] = q_geo.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.k_geo.weight"] = k_geo.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.v_proj.weight"] = teacher_v.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.out_proj.weight"] = teacher_o.to(device=device, dtype=dtype)

        # Gate (if enabled)
        if cfg.decoupled_gate:
            result[f"{attn_prefix}.gate_logit"] = torch.zeros(
                cfg.n_heads, dtype=dtype, device=device
            )

        return result

    def _get_attention_weights_copy_vo(
        self,
        teacher_weights: dict[str, torch.Tensor],
        teacher_prefix: str,
        attn_prefix: str,
        seed_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Copy teacher V/O and initialize geometric Q/K fresh.

        Useful for distillation setups where Q/K should be learned from scratch
        but we want stable V/O and residual behavior.
        """
        cfg = self.config
        result = {}

        # Semantic path: small random init
        result[f"{attn_prefix}.q_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_sem.weight"] = xavier_uniform_tensor(
            (cfg.sem_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_sem"),
            scale=cfg.sem_init_scale,
            dtype=dtype,
            device=device,
        )

        # Fresh init for geometric Q/K
        result[f"{attn_prefix}.q_geo.weight"] = xavier_uniform_tensor(
            (cfg.geo_q_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.q_geo"),
            dtype=dtype,
            device=device,
        )
        result[f"{attn_prefix}.k_geo.weight"] = xavier_uniform_tensor(
            (cfg.geo_kv_dim, cfg.d_model),
            seed=stable_hash(f"{seed_prefix}.k_geo"),
            dtype=dtype,
            device=device,
        )

        # Copy V/O from teacher
        v_key = f"{teacher_prefix}.self_attn.v_proj.weight"
        o_key = f"{teacher_prefix}.self_attn.o_proj.weight"

        for kk in (v_key, o_key):
            if kk not in teacher_weights:
                raise KeyError(f"Missing required pretrained weight: {kk}")

        teacher_v = teacher_weights[v_key]
        teacher_o = teacher_weights[o_key]

        # Validate shapes
        expected_v_shape = (cfg.v_kv_dim, cfg.d_model)
        expected_o_shape = (cfg.d_model, cfg.v_q_dim)

        if tuple(teacher_v.shape) != expected_v_shape:
            raise ValueError(
                f"v_proj shape mismatch: expected {expected_v_shape} got {tuple(teacher_v.shape)}. "
                f"Set v_head_dim={cfg.head_dim} for copy_vo mode."
            )
        if tuple(teacher_o.shape) != expected_o_shape:
            raise ValueError(
                f"o_proj shape mismatch: expected {expected_o_shape} got {tuple(teacher_o.shape)}. "
                f"Set v_head_dim={cfg.head_dim} for copy_vo mode."
            )

        result[f"{attn_prefix}.v_proj.weight"] = teacher_v.to(device=device, dtype=dtype)
        result[f"{attn_prefix}.out_proj.weight"] = teacher_o.to(device=device, dtype=dtype)

        # Gate (if enabled)
        if cfg.decoupled_gate:
            result[f"{attn_prefix}.gate_logit"] = torch.zeros(
                cfg.n_heads, dtype=dtype, device=device
            )

        return result

    def get_trainable_param_names(self, *, include_vo: bool = True) -> list[str]:
        """Get list of parameter names that should be trainable.

        Args:
            include_vo: If True, include V/O projections. Set False for copy_vo mode
                        where V/O are copied from teacher and should stay frozen.

        Returns:
            List of parameter names for attention-related weights.
        """
        trainable = []

        for i in range(self.config.n_layers):
            prefix = f"layers.{i}.attention"

            # Semantic path
            trainable.append(f"{prefix}.q_sem.weight")
            trainable.append(f"{prefix}.k_sem.weight")

            # Geometric path
            trainable.append(f"{prefix}.q_geo.weight")
            trainable.append(f"{prefix}.k_geo.weight")

            # V/O (optionally trainable)
            if include_vo:
                trainable.append(f"{prefix}.v_proj.weight")
                trainable.append(f"{prefix}.out_proj.weight")

            # Gate
            if self.config.decoupled_gate:
                trainable.append(f"{prefix}.gate_logit")

        return trainable

    def freeze_non_attention(
        self,
        model: nn.Module,
        *,
        include_vo: bool = True,
    ) -> None:
        """Freeze all parameters except attention layers.

        Args:
            model: The model to freeze
            include_vo: If True, keep V/O trainable. Set False to freeze V/O too.
        """
        trainable_names = set(self.get_trainable_param_names(include_vo=include_vo))

        for name, param in model.named_parameters():
            param.requires_grad = name in trainable_names


def run_surgery(
    teacher_weights_path: str | Path,
    student_state_dict: dict[str, torch.Tensor],
    *,
    d_model: int = 2048,
    n_layers: int = 16,
    n_heads: int = 32,
    n_kv_heads: int | None = None,
    d_ff: int = 8192,
    vocab_size: int = 128256,
    head_dim: int | None = None,
    sem_head_dim: int = 8,
    geo_head_dim: int = 32,
    v_head_dim: int | None = None,
    init_mode: str = "fresh",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convenience function to run attention surgery.

    Args:
        teacher_weights_path: Path to pretrained teacher weights
        student_state_dict: State dict of the DBA student model
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (for GQA, defaults to n_heads)
        d_ff: FFN intermediate dimension
        vocab_size: Vocabulary size
        head_dim: Teacher head dim (defaults to d_model // n_heads)
        sem_head_dim: Semantic head dim
        geo_head_dim: Geometric head dim
        v_head_dim: Value head dim (defaults to sem + geo)
        init_mode: Initialization mode (fresh, copy_vo, copy_vo_compress_qk, copy_qkvo)
        device: Device to use
        dtype: Data type to use

    Returns:
        Updated state dict with surgery applied
    """
    config = SurgeryConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        head_dim=head_dim,
        sem_head_dim=sem_head_dim,
        geo_head_dim=geo_head_dim,
        v_head_dim=v_head_dim,
    )

    surgery = AttentionSurgeryTorch(config)
    teacher_weights = surgery.load_teacher_weights(teacher_weights_path, device=device)

    return surgery.apply_surgery(
        student_state_dict,
        teacher_weights,
        init_mode=init_mode,
        device=device,
        dtype=dtype,
    )
