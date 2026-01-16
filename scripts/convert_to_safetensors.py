#!/usr/bin/env python3
"""Convert trained DBA checkpoint to SafeTensors format.

This script takes a trained DBA checkpoint (.npz) and the original Llama weights,
merges them into a complete model, and saves as SafeTensors for use with
standard inference frameworks.

Usage:
    python scripts/convert_to_safetensors.py \
        --checkpoint checkpoints/checkpoint_5000.npz \
        --output models/dba_llama_1b.safetensors

    # With custom teacher weights path:
    python scripts/convert_to_safetensors.py \
        --checkpoint checkpoints/checkpoint_5000.npz \
        --teacher /path/to/llama/model.safetensors \
        --output models/dba_llama_1b.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file as save_safetensors
from huggingface_hub import hf_hub_download, snapshot_download


def load_llama_weights(weights_path: str | Path) -> dict[str, np.ndarray]:
    """Load Llama weights from SafeTensors file."""
    from safetensors import safe_open

    weights = {}
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def load_dba_checkpoint(checkpoint_path: str | Path) -> dict[str, np.ndarray]:
    """Load DBA checkpoint from MLX .npz file."""
    data = dict(np.load(str(checkpoint_path)))
    return data


def get_llama_weights_path(model_id: str = "meta-llama/Llama-3.2-1B") -> Path:
    """Download and return path to Llama weights."""
    # Download the model files
    cache_dir = snapshot_download(model_id)
    weights_path = Path(cache_dir) / "model.safetensors"

    if not weights_path.exists():
        # Some models split weights across multiple files
        # Try to find the main weights file
        for f in Path(cache_dir).glob("*.safetensors"):
            return f
        raise FileNotFoundError(f"No safetensors file found in {cache_dir}")

    return weights_path


def convert_dba_to_llama_format(
    dba_weights: dict[str, np.ndarray],
    n_layers: int = 16,
) -> dict[str, np.ndarray]:
    """Convert DBA checkpoint keys to Llama-compatible format.

    DBA has separate Q projections for semantic and geometric paths.
    We need to map these back to a format that standard inference can use.

    DBA structure (per layer):
        - q_sem: (d_model, n_heads * sem_dim)
        - k_sem: (d_model, n_kv_heads * sem_dim)
        - q_geo: (d_model, n_heads * geo_dim)
        - k_geo: (d_model, n_kv_heads * geo_dim)
        - v_proj: (d_model, n_kv_heads * v_dim)
        - out_proj: (n_heads * v_dim, d_model)
        - decoupled_gate: (n_heads,)

    Standard Llama structure (per layer):
        - q_proj: (d_model, n_heads * head_dim)
        - k_proj: (d_model, n_kv_heads * head_dim)
        - v_proj: (d_model, n_kv_heads * head_dim)
        - o_proj: (n_heads * head_dim, d_model)
    """
    converted = {}

    for key, value in dba_weights.items():
        # Parse the key to extract layer index and param name
        # Keys look like: "layers.0.attention.q_sem.weight"
        parts = key.split(".")

        if "attention" not in key:
            # Non-attention weights pass through unchanged
            converted[key] = value
            continue

        # For now, we'll include all DBA weights with their original names
        # A full conversion would need to merge q_sem/q_geo into a single q_proj
        # but that requires understanding the DBA computation
        converted[key] = value

    return converted


def merge_weights(
    llama_weights: dict[str, np.ndarray],
    dba_weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge Llama base weights with trained DBA attention weights.

    The DBA checkpoint only contains attention parameters.
    We need to combine these with the FFN/embedding weights from Llama.
    """
    merged = {}

    # Start with all Llama weights
    merged.update(llama_weights)

    # Override/add DBA attention weights
    # The DBA checkpoint uses a different naming scheme, so we need to map
    for dba_key, dba_value in dba_weights.items():
        # Convert DBA key format to Llama key format
        # DBA: "layers.0.attention.q_sem.weight"
        # Llama: "model.layers.0.self_attn.q_proj.weight"

        if "layers." in dba_key and "attention" in dba_key:
            # Extract layer number
            parts = dba_key.split(".")
            layer_idx = parts[1]
            param_parts = parts[3:]  # Everything after "attention"
            param_name = ".".join(param_parts)

            # Create Llama-style key (but with DBA-specific names)
            llama_key = f"model.layers.{layer_idx}.self_attn.{param_name}"
            merged[llama_key] = dba_value
        else:
            # Keep other keys as-is
            merged[dba_key] = dba_value

    return merged


def save_full_model(
    merged_weights: dict[str, np.ndarray],
    output_path: str | Path,
    model_config: dict | None = None,
) -> None:
    """Save the complete model as SafeTensors with config."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save weights
    save_safetensors(merged_weights, str(output_path))
    print(f"Saved model weights to: {output_path}")

    # Save config if provided
    if model_config:
        config_path = output_path.parent / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved config to: {config_path}")


def export_dba_only(
    dba_weights: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Export just the DBA attention weights as SafeTensors.

    This is useful for loading into a custom DBA inference implementation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_safetensors(dba_weights, str(output_path))
    print(f"Saved DBA weights to: {output_path}")
    print(f"Keys: {list(dba_weights.keys())[:10]}...")  # Show first 10 keys


def main():
    parser = argparse.ArgumentParser(
        description="Convert trained DBA checkpoint to SafeTensors format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained DBA checkpoint (.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for SafeTensors file",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Path to teacher Llama weights (downloads from HF if not specified)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID for teacher weights",
    )
    parser.add_argument(
        "--dba-only",
        action="store_true",
        help="Export only DBA weights (not merged with Llama)",
    )
    args = parser.parse_args()

    # Load DBA checkpoint
    print(f"Loading DBA checkpoint from: {args.checkpoint}")
    dba_weights = load_dba_checkpoint(args.checkpoint)
    print(f"Loaded {len(dba_weights)} DBA weight tensors")

    if args.dba_only:
        # Export just the DBA weights
        export_dba_only(dba_weights, args.output)
        return

    # Get teacher weights
    if args.teacher:
        teacher_path = Path(args.teacher)
    else:
        print(f"Downloading teacher weights from: {args.model_id}")
        teacher_path = get_llama_weights_path(args.model_id)

    print(f"Loading teacher weights from: {teacher_path}")
    llama_weights = load_llama_weights(teacher_path)
    print(f"Loaded {len(llama_weights)} Llama weight tensors")

    # Merge weights
    print("Merging weights...")
    merged = merge_weights(llama_weights, dba_weights)
    print(f"Merged model has {len(merged)} tensors")

    # Save
    save_full_model(merged, args.output)

    print("\nDone! You can now load this model with:")
    print(f"  from safetensors import safe_open")
    print(f"  with safe_open('{args.output}', framework='pt') as f:")
    print(f"      weights = {{k: f.get_tensor(k) for k in f.keys()}}")


if __name__ == "__main__":
    main()
