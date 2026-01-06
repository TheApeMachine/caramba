"""Decoupled attention variants

This package contains decoupled bottleneck attention (DBA) style layers, which
separate different kinds of information flow (semantic vs geometric) inside one
attention block.
"""
from __future__ import annotations

from .layer import DecoupledAttentionLayer

__all__ = ["DecoupledAttentionLayer"]
