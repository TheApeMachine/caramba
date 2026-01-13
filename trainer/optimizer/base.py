"""Optimizer base class

A optimizer is a component that builds an optimizer from a configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from torch import nn

from caramba.config.manifest import Manifest


class Optimizer(nn.Module):
    def __init__(self, *, manifest: Manifest):
        self.manifest = manifest

    @abstractmethod
    def forward(self, params: list[nn.Parameter]) -> None:
        raise NotImplementedError("Subclasses must implement forward()")