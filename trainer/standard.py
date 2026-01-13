"""Backwards-compatible import path for the unified trainer.

`trainer.standard` is deprecated; use `trainer.train` / `caramba.trainer.trainer:Trainer`.
"""
from __future__ import annotations

from caramba.trainer.trainer import Trainer


# Backwards-compatible name.
StandardTrainer = Trainer

__all__ = ["StandardTrainer"]

