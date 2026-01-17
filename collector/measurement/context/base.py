"""Context measurement base class"""
from __future__ import annotations
from typing import Any

from caramba.collector.measurement.base import Measurement


class ContextMeasurement(Measurement):
    """Context measurement base class"""
    def __init__(self, name: str):
        super().__init__(name)

    def collect(self, data: Any) -> None:
        """Collect data from the measurement"""
        pass

    def report(self) -> dict[str, Any]:
        """Report the measurement"""
        return {self.name: self.data}