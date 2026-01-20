#!/usr/bin/env python3
"""Simple MLX inference script for DBA models.

Loads a trained DBA checkpoint and generates text. No HuggingFace required.

Usage:
    # Interactive mode
    python scripts/infer_mlx.py --checkpoint checkpoints/checkpoint_5000.npz

    # With prompt
    python scripts/infer_mlx.py --checkpoint checkpoints/checkpoint_5000.npz \
        --prompt "The quick brown fox"

    # More control
    python scripts/infer_mlx.py --checkpoint checkpoints/checkpoint_5000.npz \
        --prompt "Once upon a time" \
        --max-tokens 100 \
        --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add caramba to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter.mlx.surgery import AttentionSurgeryMLX
from layer.mlx.transformer import DBATransformer


def load_tokenizer(model_path: str | Path | None = None):
    """Load Llama tokenizer.

    Accepts either a model directory (recommended) or a weights file path
    like `.../model.safetensors` (we’ll use its parent directory).
    """
    from transformers import AutoTokenizer

    # 1. Try local path if provided
    if model_path:
        mp = Path(model_path)
        if mp.is_file():
            mp = mp.parent
        tokenizer = AutoTokenizer.from_pretrained(str(mp), use_fast=True)
        print(f"[tokenizer] Loaded from local path: {mp}")
        return tokenizer, "local"

    # 2. Try default HF hub path
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_fast=True,
        trust_remote_code=False,
    )
    print("[tokenizer] Loaded from HuggingFace: meta-llama/Llama-3.2-1B")
    return tokenizer, "llama"


def get_llama_weights_path() -> Path:
    """Find cached Llama weights."""
    # Check common cache locations
    cache_dirs = [
        Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B",
        Path.home()
        / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct",
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            # Find the safetensors file in snapshots
            for snapshot_dir in cache_dir.glob("snapshots/*"):
                weights_file = snapshot_dir / "model.safetensors"
                if weights_file.exists():
                    return weights_file

    raise FileNotFoundError(
        "Llama weights not found in cache. Run `make surgery` first to download them, "
        "or specify --teacher-weights path."
    )


def load_model(
    checkpoint_path: str | Path,
    teacher_weights_path: str | Path | None = None,
    *,
    sem_head_dim: int = 8,
    geo_head_dim: int = 32,
    v_head_dim: int | None = None,
    init_mode: str = "fresh",
) -> DBATransformer:
    """Load the DBA model and apply attention surgery."""

    # Find teacher weights if not specified
    if teacher_weights_path is None:
        teacher_weights_path = get_llama_weights_path()

    print(f"Loading teacher weights from: {teacher_weights_path}")
    print(f"Loading DBA checkpoint from: {checkpoint_path}")

    # Create surgery adapter
    surgery = AttentionSurgeryMLX(
        sem_head_dim=sem_head_dim,
    )

    # Load teacher weights
    teacher_weights = surgery.load_llama_weights(teacher_weights_path)

    # Load config.json to get RoPE scaling and tied embeddings
    teacher_path = Path(teacher_weights_path)
    teacher_dir = teacher_path if teacher_path.is_dir() else teacher_path.parent

    teacher_config_path = teacher_dir / "config.json"
    rope_scaling = None
    tie_embeddings = False

    if teacher_config_path.exists():
        with open(teacher_config_path, "r") as f:
            teacher_config = json.load(f)
            rope_scaling = teacher_config.get("rope_scaling")
            tie_embeddings = teacher_config.get("tie_word_embeddings", False)

            if rope_scaling:
                print(f"Loaded rope_scaling from {teacher_config_path}: {rope_scaling}")
            if tie_embeddings:
                print(f"Using tied embeddings (from {teacher_config_path})")

    if init_mode != "fresh" and v_head_dim is None:
        # Teacher-copy init modes require v_head_dim=head_dim (64 for Llama-3.2-1B).
        v_head_dim = 64

    # Create model
    model = surgery.create_dba_model(
        rope_scaling=rope_scaling,
        tie_embeddings=tie_embeddings,
        geo_head_dim=geo_head_dim,
        v_head_dim=v_head_dim,
    )

    # Apply surgery: copy non-attention weights, init attention per init_mode.
    model = surgery.apply_surgery(model, teacher_weights, init_mode=init_mode)

    # Load trained attention weights
    checkpoint = dict(np.load(str(checkpoint_path)))

    # Convert to MLX arrays
    trained_params = {k: mx.array(v) for k, v in checkpoint.items()}

    # Update model with trained attention weights
    # The checkpoint only contains attention params, we need to merge with the model
    current_params = model.parameters()

    # Deep merge the trained params into model
    _skipped: list[str] = []

    def merge_params(current, trained, prefix=""):
        if isinstance(current, dict):
            result = {}
            for k, v in current.items():
                key = f"{prefix}.{k}" if prefix else k
                if key in trained:
                    tv = trained[key]
                    # Only accept if shapes match (prevents loading incompatible old checkpoints)
                    if (
                        isinstance(v, mx.array)
                        and isinstance(tv, mx.array)
                        and tuple(v.shape) != tuple(tv.shape)
                    ):
                        _skipped.append(
                            f"{key} expected={tuple(v.shape)} got={tuple(tv.shape)}"
                        )
                        result[k] = v
                    else:
                        result[k] = tv
                elif isinstance(v, (dict, list)):
                    result[k] = merge_params(v, trained, key)
                else:
                    result[k] = v
            return result
        elif isinstance(current, list):
            return [
                merge_params(v, trained, f"{prefix}.{i}") for i, v in enumerate(current)
            ]
        else:
            key = prefix
            tv = trained.get(key, current)
            if (
                isinstance(current, mx.array)
                and isinstance(tv, mx.array)
                and tuple(current.shape) != tuple(tv.shape)
            ):
                _skipped.append(
                    f"{key} expected={tuple(current.shape)} got={tuple(tv.shape)}"
                )
                return current
            return tv

    # Merge trained (flat dotted) params into the nested parameter tree
    merged = merge_params(current_params, trained_params)
    if _skipped:
        print(
            "WARNING: skipped incompatible checkpoint tensors (likely an old checkpoint format):"
        )
        for s in _skipped[:20]:
            print(f"  - {s}")
        if len(_skipped) > 20:
            print(f"  ... and {len(_skipped) - 20} more")

    # Update model
    # Type narrowing: merge_params returns dict when given dict input (current_params is always dict)
    assert isinstance(merged, dict), "merge_params should return dict for dict input"
    model.update(merged)

    return model


def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample a token from logits.

    Important: sample from *logits* (not probabilities), since mlx.random.categorical
    expects unnormalized scores.
    """
    # Get logits for last position
    logits = logits[0, -1, :]  # (vocab_size,)

    # Greedy
    if temperature == 0:
        return int(mx.argmax(logits))

    # Apply temperature
    logits = logits / float(temperature)

    # Apply top-k
    if top_k is not None and top_k > 0:
        sorted_logits = mx.sort(logits)[::-1]
        threshold = sorted_logits[min(top_k - 1, len(sorted_logits) - 1)]
        logits = mx.where(logits >= threshold, logits, mx.array(float("-inf")))

    # Apply top-p (nucleus sampling)
    if top_p is not None and 0 < top_p < 1:
        sorted_indices = mx.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = mx.softmax(sorted_logits, axis=-1)
        cumsum = mx.cumsum(probs, axis=-1)

        # Keep tokens up to cumulative mass top_p (always keep at least 1 token)
        nucleus_mask = cumsum <= top_p
        # MLX doesn't support .at[].set() like JAX/NumPy; use concatenation instead
        nucleus_mask = mx.concatenate([mx.array([True]), nucleus_mask[1:]])

        masked_sorted_logits = mx.where(
            nucleus_mask, sorted_logits, mx.array(float("-inf"))
        )

        # Scatter back using fancy indexing (MLX doesn't support .at[].set())
        masked_logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        # MLX supports scatter via take/put pattern: rebuild array at sorted positions
        # Create output by placing masked_sorted_logits at sorted_indices positions
        masked_logits = mx.put_along_axis(
            masked_logits, sorted_indices, masked_sorted_logits, axis=0
        )
        logits = masked_logits

    return int(mx.random.categorical(logits))


def generate(
    model: DBATransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    stream: bool = True,
) -> str:
    """Generate text from a prompt."""
    # Encode prompt
    # For Llama tokenizers, including BOS tends to match reference behavior better.
    try:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    except TypeError:
        input_ids = tokenizer.encode(prompt)

    tokens = mx.array([input_ids])

    if stream:
        print(prompt, end="", flush=True)

    generated_tokens = []

    # Prefill (build KV cache)
    cache = []
    logits, cache, _ = model(tokens, cache=cache)

    for _ in range(max_new_tokens):
        # Sample next token from last-position logits
        next_token = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # DEBUG: Print token ID
        # print(f" [Token: {next_token}] ", end="", flush=True)

        generated_tokens.append(next_token)

        # Decode and print
        if stream:
            token_str = tokenizer.decode([next_token])
            print(token_str, end="", flush=True)

        # Check for EOS
        if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
            print(" [EOS]", end="", flush=True)
            break

        # Check for EOT (Llama 3 specific)
        # Llama 3 often uses 128001/128009 for breaks
        if next_token in (128001, 128009):
            print(" [EOT]", end="", flush=True)
            break

        # Decode step with KV cache: feed only the new token
        token_arr = mx.array([[next_token]])
        logits, cache, _ = model(token_arr, cache=cache)
        tokens = mx.concatenate([tokens, token_arr], axis=1)
        mx.eval(tokens)

    if stream:
        print()  # Newline at end

    # Return full generated text
    return tokenizer.decode(input_ids + generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="MLX inference for DBA models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained DBA checkpoint (.npz)",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        help="Path to teacher Llama weights (auto-detected if not specified)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt (interactive mode if not specified)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (None to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (omit/None to disable)",
    )
    parser.add_argument(
        "--sem-head-dim",
        type=int,
        default=8,
        help="Semantic per-head dimension (compressed content path)",
    )
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer, tok_type = load_tokenizer()
    print(f"Using tokenizer: {tok_type}")

    if tok_type == "tiktoken":
        print("WARNING: Using GPT-2 tokenizer as fallback. Output may be garbled")
        print("         since model was trained with Llama tokenizer.")
        print(
            "         Install transformers and authenticate with HuggingFace for better results."
        )
        print()

    # Load models
    print("Loading DBA student model...")
    student = load_model(
        args.checkpoint,
        args.teacher_weights,
        sem_head_dim=args.sem_head_dim,
    )

    print("Student model loaded!")

    print()

    def run_generation(prompt: str):
        generate(
            student,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    if args.prompt:
        # Single generation
        run_generation(args.prompt)
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.")
        print("-" * 50)

        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt.strip():
                    continue

                print("\nGenerating...")
                run_generation(prompt)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except EOFError:
                break

        print("\nGoodbye!")


if __name__ == "__main__":
    main()
