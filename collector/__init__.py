"""Collector module which holds objects that collect data

Primarily used for the collection of statistics and metrics
during training and validation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Collector(ABC):
    pass