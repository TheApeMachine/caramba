#!/usr/bin/env python3
"""CLI entry point for the Colab runner.

Usage:
    python -m caramba.research.dba.colab_runner --checkpoint-dir "/DBA/checkpoints/100k"
"""
from __future__ import annotations

import argparse
import sys

from caramba.research.dba.colab_runner.benchmark import BenchmarkConfig, BenchmarkRunner


def main() -> int:
    """Parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Run DBA benchmark on Colab")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/DBA/results")
    parser.add_argument("--tests-per-category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--webhook", type=str, default=None)
    parser.add_argument("--github-repo", type=str, default="theapemachine/caramba")
    parser.add_argument("--github-branch", type=str, default="main")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--folder-id", type=str, default=None)

    args = parser.parse_args()

    config = BenchmarkConfig(
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        tests_per_category=args.tests_per_category,
        seed=args.seed,
        webhook=args.webhook,
        github_repo=args.github_repo,
        github_branch=args.github_branch,
        folder_id=args.folder_id,
        headless=args.headless,
        timeout=args.timeout,
    )

    runner = BenchmarkRunner(config)
    runner.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
