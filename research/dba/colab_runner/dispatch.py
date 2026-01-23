#!/usr/bin/env python3
"""CLI entry point for the Colab runner.

This file exists for backwards compatibility with Makefile targets.
Run via: python research/dba/colab_runner/dispatch.py
Or via:  python -m caramba.research.dba.colab_runner
"""
from research.dba.colab_runner.__main__ import main

if __name__ == "__main__":
    main()
