"""Dataset base class

Defines the common interface that all datasets must implement, ensuring
consistent behavior across different data sources and making it easy to swap
datasets without changing training code.
"""
from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod


class Dataset(ABC):
    """Dataset interface

    Abstract base class that enforces a standard dataset contract: datasets
    must be indexable and have a length. This allows PyTorch DataLoaders and
    training loops to work with any dataset implementation.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize dataset

        Each dataset implementation defines its own initialization parameters,
        but all must follow this signature to work with the framework's dataset
        factory patterns.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length

        Returns the number of samples available, enabling DataLoaders to
        determine batch boundaries and progress tracking.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get sample by index

        Returns a single sample at the given index, allowing DataLoaders to
        fetch data efficiently through random or sequential access patterns.
        """
        pass