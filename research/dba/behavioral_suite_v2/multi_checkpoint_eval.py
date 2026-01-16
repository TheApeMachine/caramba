#!/usr/bin/env python3
"""
Multi-checkpoint behavioral evaluation runner.

Runs behavioral evaluation across all checkpoints in a directory,
supporting various model architectures (baseline, decoupled, gqa, bottleneck).

Usage:
    # Run on all checkpoints in 10k_runs
    python -m behavioral_suite_v2.multi_checkpoint_eval \
        --checkpoints-dir research/dba/10k_runs \
        --output-dir results/10k_behavioral

    # Run via make benchmark
    make benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class ModelSpec:
    """Specification for a model checkpoint."""
    name: str
    checkpoint_path: Path
    model_type: str  # baseline, decoupled, gqa, bottleneck
    seed: int
    variant: str = ""  # e.g., "lr2e4" for learning rate variant


def parse_checkpoint_name(path: Path) -> ModelSpec | None:
    """
    Parse a checkpoint directory name to extract model specification.

    Expected format: mac_fw1b_l12_{type}[_variant]_s{seed}
    Examples:
        mac_fw1b_l12_baseline_s1337
        mac_fw1b_l12_decoupled_s1338
        mac_fw1b_l12_baseline_lr2e4_s1337
    """
    name = path.name

    # Pattern: mac_fw1b_l12_{type}[_variant]_s{seed}
    pattern = r"mac_fw1b_l(\d+)_(\w+?)(?:_([a-z0-9]+))?_s(\d+)"
    match = re.match(pattern, name)

    if not match:
        return None

    n_layers = int(match.group(1))
    model_type = match.group(2)
    variant = match.group(3) or ""
    seed = int(match.group(4))

    # Handle variant embedded in type (e.g., "baseline_lr2e4" parsed as type="baseline", variant="lr2e4")
    if variant and model_type not in ("baseline", "decoupled", "gqa", "bottleneck"):
        # This shouldn't happen with current naming convention
        return None

    # Look for checkpoint file
    ckpt_path = path / "train_standard_final.pt"
    if not ckpt_path.exists():
        return None

    # Generate display name
    display_name = f"{model_type}"
    if variant:
        display_name = f"{model_type}_{variant}"
    display_name = f"{display_name}_s{seed}"

    return ModelSpec(
        name=display_name,
        checkpoint_path=ckpt_path,
        model_type=model_type,
        seed=seed,
        variant=variant,
    )


def parse_checkpoint_file(path: Path) -> ModelSpec | None:
    """
    Parse a checkpoint file name directly (for 100k checkpoints).

    Expected formats:
        a100_fw1b_l22_baseline_s42_100k.pt
        a100_fw1b_l22_dba_s42_100k.pt

    The model type will be inferred from the checkpoint weights.
    """
    name = path.stem  # Remove .pt extension

    # Pattern: a100_fw1b_l{layers}_{type}_s{seed}_{steps}
    patterns = [
        r"a100_fw1b_l(\d+)_(baseline|dba|decoupled|gqa|bottleneck)_s(\d+)_(\d+k?)",
        r".*_(baseline|dba|decoupled|gqa|bottleneck).*_s(\d+)",
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            groups = match.groups()
            if len(groups) >= 4:
                n_layers = int(groups[0])
                model_type = groups[1]
                seed = int(groups[2])
                steps = groups[3]
            elif len(groups) >= 2:
                model_type = groups[0]
                seed = int(groups[1])
                steps = "100k"
            else:
                continue

            # Normalize model type
            if model_type == "dba":
                model_type = "decoupled"

            display_name = f"{model_type}_s{seed}_{steps}"

            return ModelSpec(
                name=display_name,
                checkpoint_path=path,
                model_type=model_type,
                seed=seed,
                variant=steps,
            )

    return None


def discover_checkpoints(checkpoints_dir: Path) -> list[ModelSpec]:
    """Discover all valid checkpoints in a directory.

    Supports both:
    - Directory structure: mac_fw1b_l12_*/train_standard_final.pt
    - Direct .pt files: a100_fw1b_l22_baseline_s42_100k.pt
    """
    specs = []

    for item in sorted(checkpoints_dir.iterdir()):
        if item.name.startswith("."):
            continue

        # Try direct .pt file first (100k checkpoint format)
        if item.is_file() and item.suffix == ".pt":
            spec = parse_checkpoint_file(item)
            if spec:
                specs.append(spec)
            continue

        # Then try directory format (10k checkpoint format)
        if not item.is_dir():
            continue

        spec = parse_checkpoint_name(item)
        if spec:
            specs.append(spec)

    return specs


def infer_model_dims_from_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    Infer model dimensions from checkpoint state_dict.

    This allows loading any checkpoint without hardcoded dimensions.
    Returns a dict with: model_type, d_model, n_layers, d_ff, vocab_size,
    and type-specific dims (n_heads, n_kv_heads, sem_dim, geo_dim, attn_dim).
    """
    # Strip _orig_mod. prefix if present
    sd = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        sd[key] = v

    # Get basic dims from embedding
    emb = sd["embedder.token_embedding.weight"]
    vocab_size = emb.shape[0]
    d_model = emb.shape[1]

    # Count layers by finding max layer index
    n_layers = 0
    for k in sd.keys():
        if k.startswith("topology.layers.0.layers."):
            parts = k.split(".")
            if len(parts) > 4 and parts[4].isdigit():
                layer_idx = int(parts[4])
                # Each transformer block has 2 sub-layers (attn, ffn)
                # So actual layer count is max_idx // 2 + 1
                n_layers = max(n_layers, layer_idx // 2 + 1)

    # Get d_ff from SwiGLU layer (w_gate_up has shape [2*d_ff, d_model])
    ffn_key = "topology.layers.0.layers.1.layers.1.w_gate_up.weight"
    if ffn_key in sd:
        d_ff = sd[ffn_key].shape[0] // 2
    else:
        d_ff = None

    # Determine model type and attention-specific dims from first attention layer
    attn_prefix = "topology.layers.0.layers.0.layers.1"

    # Check for decoupled (has q_sem, k_sem, q_geo, k_geo)
    if f"{attn_prefix}.q_sem.weight" in sd:
        model_type = "decoupled"
        sem_dim = sd[f"{attn_prefix}.q_sem.weight"].shape[0]
        geo_dim = sd[f"{attn_prefix}.q_geo.weight"].shape[0]
        attn_dim = sd[f"{attn_prefix}.v_proj.weight"].shape[0]
        # n_heads can be inferred from out_proj: [d_model, attn_dim]
        # But we need head_dim. For decoupled, attn_dim = n_heads * head_dim
        # We can get n_heads from sem_dim if we know sem_per_head (commonly 8)
        # Better: just store attn_dim and let config figure out n_heads
        # Actually, we need n_heads for the config. Let's derive from common patterns:
        # sem_dim / n_heads = sem_per_head (typically 8)
        # geo_dim / n_heads = geo_per_head (typically 32)
        # So n_heads = sem_dim / 8 = geo_dim / 32
        n_heads = geo_dim // 32  # Most reliable
        if sem_dim // n_heads != 8:
            # Try alternate: sem_per_head = 16 (used in 100k runs)
            n_heads_alt = sem_dim // 16
            if geo_dim // n_heads_alt == 32:
                n_heads = n_heads_alt
        return {
            "model_type": model_type,
            "d_model": d_model,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "vocab_size": vocab_size,
            "n_heads": n_heads,
            "sem_dim": sem_dim,
            "geo_dim": geo_dim,
            "attn_dim": attn_dim,
        }

    # Check for standard/GQA/bottleneck (has qkv_proj)
    qkv_key = f"{attn_prefix}.qkv_proj.weight"
    out_key = f"{attn_prefix}.out_proj.weight"

    if qkv_key in sd:
        qkv_shape = sd[qkv_key].shape  # [qkv_dim, d_model]
        out_shape = sd[out_key].shape  # [d_model, attn_dim]

        qkv_dim = qkv_shape[0]
        attn_dim = out_shape[1]

        # Standard attention: qkv_dim = 3 * d_model, attn_dim = d_model
        # GQA: qkv_dim = d_model + 2 * n_kv_heads * head_dim, attn_dim = d_model
        # Bottleneck: qkv_dim = 3 * attn_dim, attn_dim < d_model

        if attn_dim == d_model and qkv_dim == 3 * d_model:
            # Standard attention
            model_type = "baseline"
            # n_heads: need to infer head_dim. Common values: 64, 128
            # For d_model=2048: n_heads=32 (head_dim=64) or n_heads=16 (head_dim=128)
            # We can't know for sure without more info, but 64 is more common for modern models
            n_heads = d_model // 64
            return {
                "model_type": model_type,
                "d_model": d_model,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "vocab_size": vocab_size,
                "n_heads": n_heads,
            }

        elif attn_dim == d_model and qkv_dim < 3 * d_model:
            # GQA: Q = d_model, K+V = 2 * n_kv_heads * head_dim
            # qkv_dim = d_model + 2 * n_kv_heads * head_dim
            # kv_total = qkv_dim - d_model = 2 * n_kv_heads * head_dim
            model_type = "gqa"
            kv_total = qkv_dim - d_model
            # We need head_dim to get n_kv_heads
            # Assume head_dim = 64 (common)
            head_dim = 64
            n_kv_heads = kv_total // (2 * head_dim)
            n_heads = d_model // head_dim
            return {
                "model_type": model_type,
                "d_model": d_model,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "vocab_size": vocab_size,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
            }

        elif attn_dim < d_model:
            # Bottleneck: attn_dim < d_model, qkv_dim = 3 * attn_dim
            model_type = "bottleneck"
            n_heads = attn_dim // 40  # 40 per head is common for bottleneck
            if n_heads == 0:
                n_heads = d_model // 64  # fallback
            return {
                "model_type": model_type,
                "d_model": d_model,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "vocab_size": vocab_size,
                "n_heads": n_heads,
                "attn_dim": attn_dim,
            }

    raise ValueError("Could not determine model type from checkpoint")


def get_model_config_from_dims(dims: dict[str, Any]) -> dict[str, Any]:
    """
    Build model config dict from inferred dimensions.
    """
    model_type = dims["model_type"]
    d_model = dims["d_model"]
    n_layers = dims["n_layers"]
    d_ff = dims["d_ff"]
    vocab_size = dims["vocab_size"]
    n_heads = dims["n_heads"]
    rope_base = 10000.0

    base_topology_layers = [
        {
            "type": "NestedTopology",
            "repeat": n_layers,
            "layers": [
                {
                    "type": "ResidualTopology",
                    "layers": [
                        {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                        None,  # Attention layer placeholder
                    ]
                },
                {
                    "type": "ResidualTopology",
                    "layers": [
                        {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                        {"type": "SwiGLULayer", "d_model": d_model, "d_ff": d_ff, "bias": False},
                    ]
                }
            ]
        },
        # Final layers wrapped in SequentialTopology to match checkpoint structure
        # (see x-final-layers in dba_paper_local.yml)
        {
            "type": "SequentialTopology",
            "layers": [
                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                {"type": "LinearLayer", "d_in": d_model, "d_out": vocab_size, "bias": False},
            ]
        },
    ]

    if model_type == "baseline":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "standard",
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "decoupled":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "decoupled",
            "dba_train_backend": "sdpa",
            "attn_dim": dims["attn_dim"],
            "sem_dim": dims["sem_dim"],
            "geo_dim": dims["geo_dim"],
            "rope_enabled": True,
            "rope_base": rope_base,
            "rope_semantic": False,
            "tie_qk": False,
            "null_attn": False,
            "decoupled_gate": False,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "gqa":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "gqa",
            "n_kv_heads": dims["n_kv_heads"],
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "bottleneck":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "standard",
            "attn_dim": dims["attn_dim"],
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    topology_layers = base_topology_layers.copy()
    topology_layers[0]["layers"][0]["layers"][1] = attn_config

    return {
        "type": "TransformerModel",
        "tied_embeddings": False,
        "embedder": {
            "type": "token",
            "vocab_size": vocab_size,
            "d_model": d_model,
        },
        "topology": {
            "type": "StackedTopology",
            "layers": topology_layers,
        }
    }


def get_model_config(model_type: str, n_layers: int = 12) -> dict[str, Any]:
    """
    Get model configuration for a given model type.

    DEPRECATED: Use infer_model_dims_from_checkpoint() + get_model_config_from_dims()
    for checkpoint-agnostic loading. This function is kept for backwards compatibility
    with the 10k_runs checkpoints trained with dba_paper_local.yml.
    """
    # Base dimensions from dba_paper_local.yml
    d_model = 2048
    n_heads = 32  # 32 heads with head_dim=64
    d_ff = 5632
    vocab_size = 50304
    rope_base = 10000.0

    # DBA dimensions from dba_paper_local.yml
    sem_dim = 256
    geo_dim = 1024
    attn_dim = 1280

    base_topology_layers = [
        {
            "type": "NestedTopology",
            "repeat": n_layers,
            "layers": [
                {
                    "type": "ResidualTopology",
                    "layers": [
                        {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                        None,  # Attention layer placeholder
                    ]
                },
                {
                    "type": "ResidualTopology",
                    "layers": [
                        {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                        {"type": "SwiGLULayer", "d_model": d_model, "d_ff": d_ff, "bias": False},
                    ]
                }
            ]
        },
        # Final layers wrapped in SequentialTopology to match checkpoint structure
        # (see x-final-layers in dba_paper_local.yml)
        {
            "type": "SequentialTopology",
            "layers": [
                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                {"type": "LinearLayer", "d_in": d_model, "d_out": vocab_size, "bias": False},
            ]
        },
    ]

    # Attention layer configuration based on model type
    if model_type == "baseline":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "standard",
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "decoupled":
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "decoupled",
            "dba_train_backend": "sdpa",
            "attn_dim": attn_dim,
            "sem_dim": sem_dim,
            "geo_dim": geo_dim,
            "rope_enabled": True,
            "rope_base": rope_base,
            "rope_semantic": False,
            "tie_qk": False,
            "null_attn": False,
            "decoupled_gate": False,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "gqa":
        # From dba_paper_local.yml: n_kv_heads_gqa: 4 (8:1 ratio with n_heads=32)
        # Checkpoint qkv_proj: [2560, 2048] = Q(2048) + K(256) + V(256)
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "gqa",
            "n_kv_heads": 4,  # 32:4 = 8:1 ratio from manifest
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    elif model_type == "bottleneck":
        # Bottleneck uses standard attention with reduced attn_dim
        # Checkpoint has: qkv_proj=[3840, 2048], out_proj=[2048, 1280]
        # This is standard MHA but with attn_dim < d_model (bottleneck)
        attn_config = {
            "type": "AttentionLayer",
            "d_model": d_model,
            "n_heads": n_heads,
            "mode": "standard",  # standard attention, just with smaller attn_dim
            "attn_dim": attn_dim,
            "rope_enabled": True,
            "rope_base": rope_base,
            "is_causal": True,
            "dropout_p": 0.0,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Insert attention config
    topology_layers = base_topology_layers.copy()
    topology_layers[0]["layers"][0]["layers"][1] = attn_config

    return {
        "type": "TransformerModel",
        "tied_embeddings": False,
        "embedder": {
            "type": "token",
            "vocab_size": vocab_size,
            "d_model": d_model,
        },
        "topology": {
            "type": "StackedTopology",
            "layers": topology_layers,
        }
    }


def _setup_caramba_imports():
    """Set up imports for caramba package.

    The caramba code uses internal imports like 'from caramba.console import logger',
    so we need to make the root directory importable as 'caramba'. This creates a
    package-like structure using types.ModuleType.
    """
    import sys
    import types

    caramba_root = Path(__file__).resolve().parent.parent.parent.parent

    # Add caramba root to path for direct imports
    if str(caramba_root) not in sys.path:
        sys.path.insert(0, str(caramba_root))

    # Create a fake 'caramba' package that redirects to the root
    if "caramba" not in sys.modules:
        caramba_pkg = types.ModuleType("caramba")
        caramba_pkg.__path__ = [str(caramba_root)]
        caramba_pkg.__file__ = str(caramba_root / "__init__.py")
        sys.modules["caramba"] = caramba_pkg


def load_model(
    spec: ModelSpec,
    device: str = "cuda",
    dtype: str = "float16",
) -> Any:
    """Load a model from checkpoint using caramba infrastructure.

    Infers model dimensions directly from checkpoint weights, so this works
    for any checkpoint without hardcoded dimensions.
    """
    _setup_caramba_imports()

    from caramba.model import Model
    from caramba.config.model import ModelConfig
    from caramba.compiler.lower import Lowerer
    from caramba.compiler.validate import Validator
    from caramba.carmath import weight_dtype

    # Load state dict first to infer dimensions
    raw_checkpoint = torch.load(
        spec.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Handle wrapped state dict (if saved with checkpoint wrapper)
    state_dict = raw_checkpoint
    if isinstance(state_dict, dict):
        for key in ("system_state_dict", "model_state_dict", "state_dict"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # Strip torch.compile prefix if present
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Infer model dimensions from checkpoint
    dims = infer_model_dims_from_checkpoint(state_dict)
    print(f"    Inferred: {dims['model_type']}, d_model={dims['d_model']}, "
          f"n_layers={dims['n_layers']}, n_heads={dims['n_heads']}, d_ff={dims['d_ff']}")

    # Build config from inferred dimensions
    model_config_dict = get_model_config_from_dims(dims)

    # Parse, lower, and validate config (same pattern as checkpoint_compare.py)
    cfg = ModelConfig.model_validate(model_config_dict)
    cfg = Lowerer().lower_model(cfg)
    Validator().validate_model_config(cfg)

    # Determine dtype
    torch_device = torch.device(device)
    torch_dtype = weight_dtype(torch_device, dtype if dtype != "auto" else "auto")

    # Create model
    model = Model(cfg).to(device=torch_device, dtype=torch_dtype)

    # Load weights
    result = model.load_state_dict(state_dict, strict=False)
    # result is a NamedTuple with .missing_keys and .unexpected_keys
    if result.missing_keys or result.unexpected_keys:
        print(f"    Warning: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    # Move to device and set dtype
    model = model.to(device=torch_device, dtype=torch_dtype)
    model.eval()

    return model


class MultiCheckpointModelWrapper:
    """
    Wraps a caramba Model to work with the behavioral test suite.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | torch.device = "cuda",
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate text continuation."""
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        input_ids = input_ids[-(self.max_length - max_new_tokens):]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = []

        # Tiktoken GPT-2 has 50257 valid tokens (0-50256).
        # Model vocab is padded to 50304 for HW efficiency.
        # Mask invalid token logits to -inf before sampling to make them impossible.
        valid_vocab_size = 50257

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                next_logits = logits[0, -1, :]

                # Mask padding tokens (50257-50303) to -inf before sampling
                # This makes them impossible to sample and preserves proper
                # probability distribution over valid tokens
                if next_logits.shape[0] > valid_vocab_size:
                    next_logits[valid_vocab_size:] = float('-inf')

                if temperature == 0.0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated_ids.append(next_token)

                # Stop on EOS
                eos_token = getattr(self.tokenizer, 'eos_token_id', None)
                if eos_token is not None and next_token == eos_token:
                    break

                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)

                if input_tensor.shape[1] > self.max_length:
                    input_tensor = input_tensor[:, -self.max_length:]

        # Decode
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(generated_ids)
        else:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """Get log probabilities for each choice token."""
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            last_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(last_logits, dim=-1)

        result = {}
        for choice in choices:
            if hasattr(self.tokenizer, 'encode'):
                choice_ids = self.tokenizer.encode(choice)
            else:
                choice_ids = self.tokenizer(choice, add_special_tokens=False)["input_ids"]

            if choice_ids:
                token_id = choice_ids[0]
                result[choice] = log_probs[token_id].item()

        return result

    def get_attention_weights(self, prompt: str):
        """Get attention weights (not implemented for multi-checkpoint eval)."""
        return None


def open_in_browser(filepath: Path) -> bool:
    """Open a file in the default web browser."""
    filepath = filepath.resolve()
    url = f"file://{filepath}"

    print(f"\nOpening results in browser: {url}")

    try:
        system = platform.system().lower()

        if system == "darwin":
            subprocess.run(["open", str(filepath)], check=True)
        elif system == "linux":
            try:
                subprocess.run(["xdg-open", str(filepath)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                webbrowser.open(url)
        elif system == "windows":
            os.startfile(str(filepath))
        else:
            webbrowser.open(url)

        return True

    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Please open manually: {filepath}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-checkpoint behavioral evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing checkpoint subdirectories (required unless --checkpoint-files is used)",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results/<timestamp>)",
    )

    parser.add_argument(
        "-n", "--tests-per-category",
        type=int,
        default=30,
        help="Number of tests per category (default: 30)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: cuda if available, else mps, else cpu)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type (default: float16)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser with results",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tiktoken:gpt2",
        help="Tokenizer to use (default: tiktoken:gpt2)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        help="Only evaluate checkpoints of this type (baseline, decoupled, gqa, bottleneck)",
    )

    parser.add_argument(
        "--filter-seed",
        type=int,
        default=None,
        help="Only evaluate checkpoints with this seed",
    )

    parser.add_argument(
        "--perplexity-csv",
        type=str,
        default=None,
        help="CSV file with model perplexities (columns: model,perplexity)",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock models for testing (no actual model loading). "
             "Useful for testing the pipeline without Python 3.12+ / caramba installed.",
    )

    parser.add_argument(
        "--checkpoint-files",
        type=str,
        nargs="+",
        default=None,
        help="Direct paths to checkpoint .pt files (alternative to --checkpoints-dir). "
             "Example: --checkpoint-files path/to/baseline.pt path/to/decoupled.pt",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("DBA Multi-Checkpoint Behavioral Evaluation")
    print("=" * 70)

    # Validate arguments
    if not args.checkpoint_files and not args.checkpoints_dir:
        print("ERROR: Must provide either --checkpoints-dir or --checkpoint-files")
        return 1

    # Handle direct checkpoint files or directory discovery
    specs = []

    if args.checkpoint_files:
        # Direct checkpoint file paths provided
        for ckpt_path in args.checkpoint_files:
            path = Path(ckpt_path)
            if not path.exists():
                print(f"ERROR: Checkpoint file not found: {path}")
                return 1
            spec = parse_checkpoint_file(path)
            if spec:
                specs.append(spec)
            else:
                print(f"WARNING: Could not parse checkpoint name: {path}")
                # Create a spec from the filename anyway
                name = path.stem
                specs.append(ModelSpec(
                    name=name,
                    checkpoint_path=path,
                    model_type="unknown",  # Will be inferred from weights
                    seed=42,
                ))
    else:
        # Discover from directory
        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
            return 1

        specs = discover_checkpoints(checkpoints_dir)

    # Apply filters
    if args.filter_type:
        specs = [s for s in specs if s.model_type == args.filter_type]
    if args.filter_seed:
        specs = [s for s in specs if s.seed == args.filter_seed]

    if not specs:
        print("ERROR: No valid checkpoints found")
        return 1

    print(f"\nDiscovered {len(specs)} checkpoint(s):")
    for spec in specs:
        print(f"  - {spec.name}: {spec.checkpoint_path}")

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nDevice: {device}")
    print(f"Tests per category: {args.tests_per_category}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./results") / f"behavioral_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load tokenizer (not needed for mock models)
    tokenizer = None
    if not args.use_mock:
        tokenizer_spec = args.tokenizer
        try:
            if tokenizer_spec.startswith("tiktoken:"):
                import tiktoken
                encoding = tokenizer_spec.split(":", 1)[1]
                tokenizer = tiktoken.get_encoding(encoding)
                print(f"\nLoaded tiktoken tokenizer: {encoding}")
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
                print(f"\nLoaded HuggingFace tokenizer: {tokenizer_spec}")
        except Exception as e:
            print(f"ERROR: Could not load tokenizer: {e}")
            return 1
    else:
        print("\nSkipping tokenizer load (mock mode)")

    # Import suite components
    from .generator import generate_suite
    from .runner import EvalRunner, EvalConfig
    from .visualizer import ResultsVisualizer, generate_html_report

    # Generate test suite
    print("\n--- Generating test suite ---")
    suite = generate_suite(
        seed=args.seed,
        tests_per_category=args.tests_per_category,
    )
    print(f"Generated {len(suite.tests)} tests across {len(suite.category_counts)} categories")

    if args.verbose:
        for cat, count in suite.category_counts.items():
            print(f"  - {cat}: {count}")

    # Save test suite
    suite_path = output_dir / "test_suite.json"
    with open(suite_path, 'w') as f:
        json.dump(suite.to_dict(), f, indent=2, default=str)
    print(f"Saved test suite to: {suite_path}")

    # Load and wrap models
    print("\n--- Loading models ---")
    models = {}

    if args.use_mock:
        # Use mock models for testing the pipeline
        from .runner import MockModel
        print("Using mock models (--use-mock specified)")
        for spec in specs:
            # Vary accuracy by model type for interesting results
            accuracy_map = {
                "baseline": 0.70,
                "decoupled": 0.85,
                "gqa": 0.75,
                "bottleneck": 0.80,
            }
            accuracy = accuracy_map.get(spec.model_type, 0.75)
            models[spec.name] = MockModel(spec.name, accuracy=accuracy)
            print(f"  Created mock model: {spec.name} (accuracy={accuracy})")
    else:
        for spec in specs:
            print(f"Loading {spec.name}...")
            try:
                raw_model = load_model(spec, device=device, dtype=args.dtype)
                wrapped_model = MultiCheckpointModelWrapper(
                    model=raw_model,
                    tokenizer=tokenizer,
                    device=device,
                )
                models[spec.name] = wrapped_model
                print(f"  OK: {spec.name}")
            except Exception as e:
                print(f"  FAILED: {spec.name} - {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    if not models:
        print("ERROR: No models loaded successfully")
        return 1

    print(f"\nLoaded {len(models)} model(s): {list(models.keys())}")

    # Run evaluation
    print("\n--- Running evaluation ---")
    config = EvalConfig(
        max_new_tokens=args.max_new_tokens,
        capture_attention=False,  # Skip attention for multi-model eval
        show_progress=True,
    )

    runner = EvalRunner(models, config)
    results = runner.run(suite.tests)

    # Save results
    print("\n--- Saving results ---")
    results.save(output_dir)

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    viz = ResultsVisualizer()

    # Per-category comparison
    viz.plot_category_comparison(
        results.category_results,
        metric="exact_match_rate",
        output_path=output_dir / "category_exact_match.png",
    )

    viz.plot_category_comparison(
        results.category_results,
        metric="soft_score_avg",
        output_path=output_dir / "category_soft_score.png",
    )

    # Head-to-head matrix
    if len(models) > 1:
        viz.plot_head_to_head_matrix(
            results.comparisons,
            output_path=output_dir / "head_to_head.png",
        )

    # Score distribution
    viz.plot_soft_score_distribution(
        results.scores,
        output_path=output_dir / "score_distribution.png",
    )

    # Failure modes
    viz.plot_failure_modes(
        results.summaries,
        output_path=output_dir / "failure_modes.png",
    )

    # Load perplexities if provided
    perplexities = {}
    if args.perplexity_csv:
        csv_path = Path(args.perplexity_csv)
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get('model') or row.get('name') or row.get('model_name')
                    ppl = row.get('perplexity') or row.get('ppl')
                    if model and ppl:
                        try:
                            perplexities[model] = float(ppl)
                        except ValueError:
                            pass

    if perplexities:
        print(f"\nLoaded perplexity data for {len(perplexities)} model(s)")
        matched_perplexities = {}
        for model_id in results.model_ids:
            if model_id in perplexities:
                matched_perplexities[model_id] = perplexities[model_id]
            else:
                for ppl_name, ppl_val in perplexities.items():
                    if ppl_name in model_id or model_id in ppl_name:
                        matched_perplexities[model_id] = ppl_val
                        break

        if matched_perplexities:
            print(f"Generating Pareto curve with {len(matched_perplexities)} models...")
            viz.plot_pareto_from_results(
                summaries=results.summaries,
                perplexities=matched_perplexities,
                accuracy_metric="exact_match_rate",
                output_path=output_dir / "pareto_curve.png",
            )

    # Generate HTML report
    html_path = output_dir / "report.html"
    generate_html_report(results, html_path)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    # Group by model type for summary
    by_type: dict[str, list[str]] = {}
    for spec in specs:
        if spec.name in results.summaries:
            if spec.model_type not in by_type:
                by_type[spec.model_type] = []
            by_type[spec.model_type].append(spec.name)

    for model_type, names in sorted(by_type.items()):
        print(f"\n{model_type.upper()} Models:")
        for name in sorted(names):
            summary = results.summaries[name]
            print(f"  {name}:")
            print(f"    Exact match: {summary['exact_match_rate']:.1%}")
            print(f"    Content match: {summary['content_match_rate']:.1%}")
            print(f"    Soft score: {summary['soft_score_avg']:.2f}")

    # Cross-type comparison
    if len(by_type) > 1:
        print("\n" + "-" * 40)
        print("CROSS-TYPE COMPARISON (average across seeds)")
        print("-" * 40)

        for model_type, names in sorted(by_type.items()):
            summaries = [results.summaries[n] for n in names]
            avg_exact = sum(s['exact_match_rate'] for s in summaries) / len(summaries)
            avg_content = sum(s['content_match_rate'] for s in summaries) / len(summaries)
            avg_soft = sum(s['soft_score_avg'] for s in summaries) / len(summaries)
            print(f"  {model_type:12}: exact={avg_exact:.1%}, content={avg_content:.1%}, soft={avg_soft:.2f}")

    print(f"\nResults saved to: {output_dir}")
    print(f"HTML report: {html_path}")

    # Open in browser
    if not args.no_browser:
        open_in_browser(html_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
