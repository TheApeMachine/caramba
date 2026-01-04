"""MOSAIC block package

Exports the MOSAIC block layer as a stable symbol for manifest-driven topologies.
This package exists to keep the implementation split into small, composable parts.
"""

from caramba.layer.mosaic.block.layer import MosaicBlockLayer

__all__ = ["MosaicBlockLayer"]

