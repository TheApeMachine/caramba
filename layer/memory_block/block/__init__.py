"""Streaming memory block layer

This package exposes the block layer while keeping the implementation split into
small, composable submodules.
"""

from layer.memory_block.block.layer import MemoryBlockLayer

__all__ = ["MemoryBlockLayer"]

