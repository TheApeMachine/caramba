"""Context builder"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.context.base import BaseContext
from caramba.trainer.context.run import RunCtx
from caramba.manifest.target import TargetType

class ContextBuilder:
    def __init__(self, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    def build(self) -> BaseContext:
        if self.target.type == TargetType.EXPERIMENT:
            return RunCtx().from_target(self.target)
        else:
            raise ValueError(f"Unsupported target type: {self.target.type}")