"""Compatibility shim for Mosaic StateBank.

`StateBank` moved to `caramba.layer.memory_block.block.state_bank`, but some code and
configs still import it from `caramba.layer.memory_block.state_bank`.
"""

from __future__ import annotations

from caramba.layer.memory_block.block.state_bank import StateBank

__all__ = ["StateBank"]

