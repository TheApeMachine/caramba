"""Checkpointer builder

Builds a checkpointer based on the manifest.
"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.checkpointer.base import CheckPointer
from caramba.trainer.checkpointer.phase import PhaseCheckPointer
from caramba.trainer.context.run import RunCtx


class CheckpointerBuilder:
    """Builder for checkpointer components."""

    def __init__(self, ctx: RunCtx, manifest: Manifest, target: Target) -> None:
        """Initialize the checkpointer builder.

        Args:
            manifest: The manifest configuration
            target: The target configuration
        """
        self.ctx = ctx
        self.manifest = manifest
        self.target = target

    def build(self) -> CheckPointer:
        """Build and return a checkpointer instance."""
        return PhaseCheckPointer(
            ctx=self.ctx, manifest=self.manifest, target=self.target
        )