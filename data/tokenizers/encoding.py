"""Tokenization outputs used by training tokenizers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Encoding:
    """Minimal encoding object returned by training tokenizers."""

    ids: list[int]

