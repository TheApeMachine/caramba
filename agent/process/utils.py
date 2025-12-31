"""Shared utilities for agent process implementations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from caramba.config.manifest import Manifest


def _manifest_root_dir(*, manifest: Manifest, manifest_path: Path | None) -> Path:
    """Return the root artifacts directory for a manifest."""
    name = str(manifest.name or (manifest_path.stem if manifest_path else "manifest"))
    return Path("artifacts") / name


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    s = text.strip()
    if not s:
        return None

    # Fast-path: whole-string JSON.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Heuristic: locate the first {...} block.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj2 = json.loads(m.group(0))
        return obj2 if isinstance(obj2, dict) else None
    except Exception:
        return None

