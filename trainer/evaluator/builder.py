"""Evaluator builder

Builds an evaluator based on the manifest.
"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.manifest.evaluator import EvaluatorType
from caramba.trainer.evaluator.base import Evaluator
from caramba.trainer.evaluator.default import DefaultEvaluator

class EvaluatorBuilder:
    def __init__(self, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    def build(self) -> Evaluator:
        match self.target.evaluator.type:
            case EvaluatorType.DEFAULT:
                return DefaultEvaluator(manifest=self.manifest, target=self.target)
            case _:
                raise ValueError(f"Unsupported evaluator type: {self.target.evaluator.type}")
