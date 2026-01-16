#!/usr/bin/env python3
"""
Convenience script to run the DBA Behavioral Test Suite.

Usage:
    # Run with caramba manifest (recommended)
    python research/dba/run_eval.py --manifest research/dba/benchmark.yml

    # Use mock models for testing the pipeline
    python research/dba/run_eval.py --use-mock -n 10

    # Full options
    python research/dba/run_eval.py --help
"""
import sys
from pathlib import Path

# Add the behavioral_suite_v2 directory to path
suite_dir = Path(__file__).parent / "behavioral_suite_v2"
if str(suite_dir.parent) not in sys.path:
    sys.path.insert(0, str(suite_dir.parent))

from behavioral_suite_v2.cli import main

if __name__ == "__main__":
    sys.exit(main())
