"""Hugging Face Hub integration for downloading model checkpoints.

This module provides backward compatibility for the HFLoader alias.
New code should use caramba.loader.checkpoint.hf.HFCheckpoint directly.
"""
from __future__ import annotations

from loader.checkpoint.hf import HFCheckpoint

# Backward compatibility alias
HFLoader = HFCheckpoint

__all__ = ["HFLoader"]
