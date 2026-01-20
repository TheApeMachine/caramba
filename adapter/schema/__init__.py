"""Adapter schemas.

Schemas describe external naming/layout conventions as data. Adapter
implementations consume schemas to avoid schema-as-code duplication.
"""

from __future__ import annotations

from adapter.schema.base import StateDictSchema
from adapter.schema.loader import SchemaLoader

__all__ = ["StateDictSchema", "SchemaLoader"]

