"""Accuracy benchmark utility functions

Encapsulated as specializations to avoid loose helpers and enable composition.
"""
from __future__ import annotations

from typing import Any


class TextNormalization:
    """Text normalization behaviors used by benchmarks."""

    @staticmethod
    def norm_ws(s: str) -> str:
        return " ".join(str(s).replace("\r\n", "\n").replace("\r", "\n").split())


class DictCoercion:
    """Coercion behaviors for external dataset row types."""

    @staticmethod
    def as_dict(x: Any) -> dict[str, Any]:
        """Conversion from dataset row to dict (dict rows only)."""
        return x if isinstance(x, dict) else {}
