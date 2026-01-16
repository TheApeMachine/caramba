#!/usr/bin/env python3
"""CLI for running the MLX routing hypothesis experiment.

Usage:
    python -m caramba.cli.mlx_routing \\
        --weights path/to/llama_weights.safetensors \\
        --data path/to/tokens.npy \\
        --steps 5000

This runs attention surgery on a pretrained Llama model using MLX,
replacing attention with fresh DBA layers and training only those
layers while keeping FFN/embeddings frozen.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLX routing hypothesis experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to pretrained Llama weights (.safetensors or .npz)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to tokenized training data (.npy)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Context length (default: 2048)",
    )
    parser.add_argument(
        "--sem-dim",
        type=int,
        default=256,
        help="Semantic dimension (default: 256 = 8 dims/head)",
    )
    parser.add_argument(
        "--geo-dim",
        type=int,
        default=512,
        help="Geometric dimension (default: 512 = 16 dims/head)",
    )
    parser.add_argument(
        "--v-dim",
        type=int,
        default=768,
        help="Value dimension (default: 768 = 24 dims/head)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/mlx_routing",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="LR warmup steps (default: 500)",
    )

    args = parser.parse_args()

    # Check MLX availability
    try:
        import mlx.core as mx

        print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    except ImportError:
        print("ERROR: MLX not installed. Install with: pip install mlx", file=sys.stderr)
        sys.exit(1)

    # Import trainer
    from caramba.trainer.mlx.routing_hypothesis import run_routing_hypothesis_mlx

    # Run experiment
    run_routing_hypothesis_mlx(
        teacher_weights_path=args.weights,
        data_path=args.data,
        sem_dim=args.sem_dim,
        geo_dim=args.geo_dim,
        v_dim=args.v_dim,
        max_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        block_size=args.block_size,
        save_dir=args.save_dir,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
