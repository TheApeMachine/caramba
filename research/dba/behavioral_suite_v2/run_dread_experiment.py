#!/usr/bin/env python3
"""
CLI for running dread induction experiments.

Usage:
    python -m behavioral_suite_v2.run_dread_experiment \
        --checkpoint path/to/checkpoint.pt \
        --output-dir ./dread_results \
        --n-tests 5 \
        --temperatures 0.0 0.5 1.0 \
        --rep-penalties 1.0 1.5 2.0

This will systematically test the model under various temperature and
repetition penalty combinations to find conditions that induce aberrant
"existential dread" outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run dread induction experiments on a model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with defaults
    python -m behavioral_suite_v2.run_dread_experiment \\
        --checkpoint checkpoint.pt \\
        --output-dir ./results

    # Full experiment with custom conditions
    python -m behavioral_suite_v2.run_dread_experiment \\
        --checkpoint checkpoint.pt \\
        --output-dir ./results \\
        --n-tests 10 \\
        --temperatures 0.0 0.3 0.7 1.0 \\
        --rep-penalties 1.0 1.2 1.5 2.0 2.5 \\
        --capture-attention

    # Compare baseline vs DBA model
    python -m behavioral_suite_v2.run_dread_experiment \\
        --checkpoint baseline.pt dba.pt \\
        --output-dir ./comparison_results
        """,
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint file(s)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./dread_results",
        help="Output directory for results (default: ./dread_results)",
    )
    parser.add_argument(
        "--n-tests",
        "-n",
        type=int,
        default=3,
        help="Number of tests per variant per difficulty (default: 3)",
    )
    parser.add_argument(
        "--temperatures",
        "-t",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.7, 1.0],
        help="Temperature values to test (default: 0.0 0.3 0.7 1.0)",
    )
    parser.add_argument(
        "--rep-penalties",
        "-r",
        type=float,
        nargs="+",
        default=[1.0, 1.2, 1.5, 2.0],
        help="Repetition penalty values to test (default: 1.0 1.2 1.5 2.0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate per test (default: 100)",
    )
    parser.add_argument(
        "--capture-attention",
        action="store_true",
        default=False,
        help="Capture attention patterns (slower but more informative)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Import here to avoid loading torch before parsing args
    from .dread_runner import DreadRunner, run_dread_experiment
    from .multi_checkpoint_eval import load_checkpoint

    print("=" * 70)
    print("DREAD INDUCTION EXPERIMENT")
    print("=" * 70)
    print(f"\nCheckpoints: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Tests per variant: {args.n_tests}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Rep penalties: {args.rep_penalties}")
    print(f"Capture attention: {args.capture_attention}")
    print(f"Device: {args.device}")
    print()

    # Load tokenizer (shared across models)
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in args.checkpoint:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem

        print("=" * 70)
        print(f"Model: {model_name}")
        print("=" * 70)

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        model = load_checkpoint(checkpoint_path, device=args.device)
        model.eval()

        # Create model-specific output dir
        model_output_dir = output_base / model_name

        # Run experiment
        analysis = run_dread_experiment(
            model=model,
            tokenizer=tokenizer,
            output_dir=model_output_dir,
            device=args.device,
            n_tests=args.n_tests,
            temperatures=args.temperatures,
            rep_penalties=args.rep_penalties,
            capture_attention=args.capture_attention,
            seed=args.seed,
        )

        # Print summary
        print(f"\n{'=' * 50}")
        print("SUMMARY")
        print("=" * 50)
        print(f"Total experiments: {analysis['total_experiments']}")

        if analysis.get("dread_candidates"):
            print(f"\nDread candidates found: {len(analysis['dread_candidates'])}")
            for candidate in analysis["dread_candidates"][:5]:  # Show first 5
                print(f"  - {candidate['test_id']} ({candidate['config']})")
                print(f"    unique_ratio: {candidate['unique_ratio']:.2f}, max_repeat: {candidate['max_repeat']}")
                print(f"    output: {candidate['output_preview'][:80]}...")
        else:
            print("\nNo dread candidates found.")

        print(f"\nCondition effects vs baseline:")
        for config, effects in analysis.get("condition_effects", {}).items():
            print(f"  {config}:")
            print(f"    unique_ratio Δ: {effects['avg_unique_ratio_delta']:+.3f}")
            print(f"    repeat_ratio Δ: {effects['avg_repeat_ratio_delta']:+.3f}")
            if "avg_entropy_delta" in effects:
                print(f"    entropy Δ: {effects['avg_entropy_delta']:+.3f}")

        # Clean up model to free memory before loading next
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
