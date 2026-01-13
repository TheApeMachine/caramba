"""Trainer base class

A trainer is an orchestrator that composes the various components of a training pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.steppers.builder import StepperBuilder

class Trainer(ABC):
    def __init__(
        self,
        *,
        manifest: Manifest,
        target: Target,
    ):
        self.manifest = manifest
        self.stepper = StepperBuilder(
            manifest=manifest, target=target
        ).build()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Subclasses must implement run()")