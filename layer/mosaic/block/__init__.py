"""MOSAIC block layers

This package exposes the MOSAIC block as a stable symbol while keeping its
implementation split into small, composable submodules, so the block remains
inspectable and hackable as the design evolves.
"""

from caramba.layer.mosaic.block.layer import MosaicBlockLayer

__all__ = ["MosaicBlockLayer"]

