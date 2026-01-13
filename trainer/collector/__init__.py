"""Collector module"""
from __future__ import annotations

from caramba.trainer.collector.base import Collector
from caramba.trainer.collector.builder import CollectorBuilder

__all__ = [
    "Collector",
    "CollectorBuilder",
]