"""Entrypoints model.

Entrypoints are named shortcuts (aliases) to targets, used by the CLI when a
user passes `--target <alias>`.
"""
from __future__ import annotations

from typing import Dict

from pydantic import RootModel


class Entrypoints(RootModel[Dict[str, str]]):
    """Dictionary mapping entrypoint names to target references."""
