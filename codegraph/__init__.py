"""Static code graph parsing + FalkorDB ingestion.

This module builds a deterministic structural graph for Python code:
- modules, classes, functions, methods
- imports, inheritance, ownership, basic call edges (static approximation)
"""

from __future__ import annotations

__all__ = ["parse_repo", "sync_files_to_falkordb"]

from caramba.codegraph.parser import parse_repo
from caramba.codegraph.sync import sync_files_to_falkordb

