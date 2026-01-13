"""Checkpointer base class

Checkpointers own checkpoint naming, save/load mechanics, and resume discovery.
They are kept separate from training loops so different storage/layout policies
can be swapped without touching core trainer logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.context.base import BaseContext
from caramba.trainer.context.run import RunCtx


class CheckPointer(ABC):
    """Checkpointer interface."""
    def __init__(self, *, ctx: RunCtx, manifest: Manifest, target: Target) -> None:
        """Create a checkpointer bound to a manifest target."""
        self.ctx = ctx
        self.manifest = manifest
        self.target = target

    @abstractmethod
    def step(self) -> None:
        """Save a checkpoint and return its path."""
        raise NotImplementedError("Subclasses must implement step()")
