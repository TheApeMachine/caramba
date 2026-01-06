"""Compatibility shim for Mosaic StateBank.

`StateBank` moved to `caramba.layer.mosaic.block.state_bank`, but some code and
configs still import it from `caramba.layer.mosaic.state_bank`.
"""

from __future__ import annotations

from caramba.layer.mosaic.block.state_bank import StateBank

__all__ = ["StateBank"]

