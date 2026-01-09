"""Accuracy task"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskAccuracy:
    """Task accuracy"""
    task: str
    split: str
    accuracy: float
    correct: int
    total: int