"""Evaluator loss class"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from trainer.evaluator import Evaluator


class Loss(Evaluator):
    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    def evaluate(self) -> None:
        raise NotImplementedError("Subclasses must implement evaluate()")