"""Base class for measurements

A measurement is a single unit of data that is collected during training or
validation. It is used to track the performance of the model and the data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Measurement(ABC):
    """Base class for measurements"""
    def __init__(self, name: str):
        self.name = name
        self.data: list[Any] = []

    def collect(self, data: Any) -> None:
        """Collect data from the measurement"""
        pass

    @abstractmethod
    def report(self) -> dict[str, Any]:
        """Report the measurement"""
        return {self.name: self.data}

    def __str__(self) -> str:
        """String representation of the measurement"""
        return f"{self.name}: {self.data}"

    def __repr__(self) -> str:
        """Representation of the measurement"""
        return f"{self.name}: {self.data}"