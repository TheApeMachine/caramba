"""MOSAIC memory package

Implements the hard-addressed, set-associative memory used by MOSAIC blocks.
This package exists to keep responsibilities small, testable, and composable.
"""

from caramba.layer.mosaic.memory.memory import MosaicMemory

__all__ = ["MosaicMemory"]

