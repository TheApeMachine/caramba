"""Task accuracy measurement

Represents the accuracy performance on a specific task split.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccuracySample:
    """A single example outcome from an accuracy task."""
    prompt: str
    gold: str
    pred: str
    ok: bool

@dataclass
class TaskAccuracy:
    """Task accuracy measurement

    Attributes:
        task: The task name (e.g. 'hellaswag')
        split: The dataset split (e.g. 'validation')
        accuracy: The accuracy score (0.0 to 1.0)
        correct: Number of correct predictions
        total: Total number of examples evaluated
        samples: List of individual samples for logging
        elapsed_seconds: Time taken to run the task (for speed comparison)
    """
    task: str
    split: str
    accuracy: float
    correct: int
    total: int
    samples: list[AccuracySample] = field(default_factory=list)
    elapsed_seconds: float = 0.0
