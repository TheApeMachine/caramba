"""Graph batch transform

Placeholder for future graph batching functionality, currently a no-op
that maintains the transform interface for graph datasets.
"""
from __future__ import annotations

from dataclasses import dataclass

from caramba.runtime.tensordict_utils import TensorDictBase


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """Graph batching placeholder

    Placeholder for future graph batching functionality. Currently a no-op
    that passes data through unchanged, allowing graph datasets to work with
    the transform pipeline without special handling.
    """

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Pass through unchanged

        Returns the input TensorDict without modification, maintaining the
        transform interface while deferring graph batching implementation.
        """
        return td
