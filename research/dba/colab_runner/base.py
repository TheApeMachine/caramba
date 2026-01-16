"""Base classes for Colab runner

This module provides base classes for Colab runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class ColabRunnerBase(ABC):
    """Base class for Colab runner."""
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the Colab runner."""
        raise NotImplementedError("Subclasses must implement this method.")