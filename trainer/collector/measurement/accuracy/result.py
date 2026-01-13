"""Accuracy benchmark result

Aggregates accuracy measurements across multiple tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .task import TaskAccuracy


@dataclass
class AccuracyResult:
    """Accuracy benchmark result

    Aggregates task accuracy measurements for a model.
    """
    model_name: str
    tasks: list[TaskAccuracy] = field(default_factory=list)

    @property
    def micro_accuracy(self) -> float:
        """Overall accuracy weighted by number of examples."""
        tot = sum(int(t.total) for t in self.tasks)
        if tot <= 0:
            return 0.0
        cor = sum(int(t.correct) for t in self.tasks)
        return float(cor) / float(tot)
