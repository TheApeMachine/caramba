"""Manifest module

The manifest is the single source of truth that governs the entire
caramba research substrate.
"""
from __future__ import annotations

from .base import BaseManifest as Manifest


__all__ = [
    "Manifest",
]
