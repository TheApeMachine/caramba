"""Collector module which holds objects that collect data

Primarily used for the collection of statistics and metrics
during training and validation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Collector(ABC):
    """Collector base class"""
    def __init__(self, name: str):
        self.name = name
        self.data: list[Any] = []

    """Collector base class"""
    @abstractmethod
    def observe(self) -> None:
        """Collect data from the collector"""
        pass

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Report the collected data"""
        pass