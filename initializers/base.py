"""Base interface for model initializers."""

from __future__ import annotations

from typing import Protocol, Any
from torch import nn

class Initializer(Protocol):
    """Protocol for components that can initialize model weights."""

    def initialize(self, module: nn.Module) -> None:
        """Initialize the weights of the given module (in-place)."""
        ...
