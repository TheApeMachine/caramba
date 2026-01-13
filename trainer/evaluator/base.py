"""Evaluator base class"""
from __future__ import annotations

from abc import ABC, abstractmethod

from caramba.manifest import Manifest
from caramba.manifest.target import Target

class Evaluator(ABC):
    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    @abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError("Subclasses must implement evaluate()")