"""Small helpers for safe, dependency-light instrumentation.

Instrumentation should never be able to break training. This module provides
best-effort utilities (time, JSON coercion) that avoid raising exceptions and
keep the caller logic clean.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from console import logger


def now_s() -> float:
    """Return current wall-clock time as seconds since epoch.

    This is used for lightweight timestamps in JSONL logs.
    """

    return float(time.time())


def _json_fallback(obj: object) -> str:
    """Fallback serializer for non-JSONable objects."""

    return repr(obj)


def coerce_jsonable(value: Any) -> Any:
    """Best-effort conversion of common numeric types to JSON-friendly values.

    Why this exists:
    - PyTorch / numpy scalars can appear in metrics dicts.
    - Tensors must not be written directly.
    - Non-finite floats should not poison downstream analysis.
    """

    # Cheap fast paths.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)

    # Tensor-like objects: avoid `.item()` because it may trigger device sync (CUDA).
    # We still keep numpy scalar support below, since numpy scalars don't have `.device`.
    try:
        dev = getattr(value, "device", None)
        if dev is not None:
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            return {
                "_type": type(value).__name__,
                "shape": list(shape) if shape is not None else None,
                "device": str(dev),
                "dtype": None if dtype is None else str(dtype),
            }
    except Exception as e:
        logger.error(f"Failed to coerce tensor-like object, continuing: {value}: {e}")

    # Avoid importing torch/numpy: rely on duck-typing for numpy scalars.
    try:
        item = getattr(value, "item", None)
        if callable(item):
            out = item()
            return coerce_jsonable(out)
    except Exception as e:
        logger.error(f"Failed to coerce item, continuing: {value}: {e}")

    # Tensors / arrays: refuse by default (too big). Caller can summarize first.
    try:
        shape = getattr(value, "shape", None)
        if shape is not None and not isinstance(value, (list, tuple, dict)):
            return {"_type": type(value).__name__, "shape": list(shape)}  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Failed to coerce shape, continuing: {value}: {e}")

    # Containers: recursively coerce.
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(v) for v in value]

    return repr(value)


def dumps_json(obj: dict[str, Any]) -> str:
    """Serialize a dict to JSON with safe fallbacks."""

    return json.dumps(obj, ensure_ascii=False, default=_json_fallback)

