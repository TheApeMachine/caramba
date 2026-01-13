"""Evaluator module"""
from __future__ import annotations

from caramba.trainer.evaluator.base import Evaluator
from caramba.trainer.evaluator.builder import EvaluatorBuilder

__all__ = [
    "Evaluator",
    "EvaluatorBuilder",
]