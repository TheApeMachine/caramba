"""Accuracy benchmark tasks package"""
from __future__ import annotations

from caramba.benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from caramba.benchmark.accuracy.tasks.builder import BenchmarkAccuracyTaskBuilder
from caramba.benchmark.accuracy.tasks.hellaswag import BenchmarkAccuracyTaskHellaswag
from caramba.benchmark.accuracy.tasks.piqa import BenchmarkAccuracyTaskPiqa
from caramba.benchmark.accuracy.tasks.winogrande import BenchmarkAccuracyTaskWinogrande
from caramba.benchmark.accuracy.tasks.arc_easy import BenchmarkAccuracyTaskArcEasy
from caramba.benchmark.accuracy.tasks.arc_challenge import BenchmarkAccuracyTaskArcChallenge
from caramba.benchmark.accuracy.tasks.boolq import BenchmarkAccuracyTaskBoolq

__all__ = [
    "BenchmarkAccuracyTask",
    "BenchmarkAccuracyTaskBuilder",
    "BenchmarkAccuracyTaskHellaswag",
    "BenchmarkAccuracyTaskPiqa",
    "BenchmarkAccuracyTaskWinogrande",
    "BenchmarkAccuracyTaskArcEasy",
    "BenchmarkAccuracyTaskArcChallenge",
    "BenchmarkAccuracyTaskBoolq",
]
