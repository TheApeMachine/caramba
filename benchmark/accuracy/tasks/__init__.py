"""Accuracy benchmark tasks package"""
from __future__ import annotations

from benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from benchmark.accuracy.tasks.builder import BenchmarkAccuracyTaskBuilder
from benchmark.accuracy.tasks.hellaswag import BenchmarkAccuracyTaskHellaswag
from benchmark.accuracy.tasks.piqa import BenchmarkAccuracyTaskPiqa
from benchmark.accuracy.tasks.winogrande import BenchmarkAccuracyTaskWinogrande
from benchmark.accuracy.tasks.arc_easy import BenchmarkAccuracyTaskArcEasy
from benchmark.accuracy.tasks.arc_challenge import BenchmarkAccuracyTaskArcChallenge
from benchmark.accuracy.tasks.boolq import BenchmarkAccuracyTaskBoolq

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
