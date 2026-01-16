#!/usr/bin/env python3
"""Generate behavioral test suite JSON file."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

from behavioral_suite_v2 import generate_suite


def main():
    parser = argparse.ArgumentParser(description="Generate behavioral test suite")
    parser.add_argument("-n", "--tests-per-category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="behavioral_results/test_suite.json")
    args = parser.parse_args()

    # Generate suite
    suite = generate_suite(seed=args.seed, tests_per_category=args.tests_per_category)
    print(f"Generated {len(suite.tests)} tests across {len(suite.category_counts)} categories")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(suite.to_dict(), f, indent=2, default=str)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
