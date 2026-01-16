#!/usr/bin/env python3
"""
Convenience script to run the behavioral evaluation suite.

Usage:
    python run_eval.py /path/to/checkpoints
    python run_eval.py /path/to/checkpoints --tests-per-category 50
    python run_eval.py /path/to/checkpoints -o ./my_results --no-browser

This script auto-discovers all model checkpoints in the given directory,
runs the full behavioral test suite, and opens the HTML dashboard.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral_suite_v2.cli import main

if __name__ == "__main__":
    sys.exit(main())
