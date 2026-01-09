"""Task accuracy measurement

Represents the accuracy performance on a specific task split.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskAccuracy:
    """Task accuracy measurement

    Attributes:
        task: The task name (e.g. 'hellaswag')
        split: The dataset split (e.g. 'validation')
        accuracy: The accuracy score (0.0 to 1.0)
        correct: Number of correct predictions
        total: Total number of examples evaluated
    """
    task: str
    split: str
    accuracy: float
    correct: int
    total: int
