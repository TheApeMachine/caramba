"""Optimizer builder

Builds an optimizer based on the manifest.
"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.manifest.optimizer import OptimizerType
from caramba.optimizer.base import Optimizer

class OptimizerBuilder:
    def __init__(self, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    def build(self) -> Optimizer:
        match self.target.optimizer.type:
            case OptimizerType.ADAM:
                return Adam(manifest=self.manifest, target=self.target)
            case OptimizerType.ADAMW:
                return AdamW(manifest=self.manifest, target=self.target)
            case OptimizerType.SGD:
                return SGD(manifest=self.manifest, target=self.target)
            case OptimizerType.LION:
                return Lion(manifest=self.manifest, target=self.target)
            case _:
                raise ValueError(f"Unsupported optimizer type: {self.target.optimizer.type}")