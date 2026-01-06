"""MOSAIC forward paths

Contains the two compute paths for MOSAIC blocks:
- FastTrainPath: chunked training path (T>1) to reduce overhead
- SequentialPath: exact streaming semantics path
"""
from __future__ import annotations

from torch import Tensor, nn

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.state_bank import StateBank
from caramba.layer.mosaic.memory import MosaicMemory


class Path(nn.Module):
    """MOSAIC forward path

    This is a base class for all MOSAIC forward paths.
    """
    def __init__(
        self,
        *,
        state: StateBank,
        memory: MosaicMemory,
        gate_long: nn.Linear,
        gate_mem: nn.Linear,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.state = state
        self.memory = memory
        self.gate_long = gate_long
        self.gate_mem = gate_mem
        self.chunk_size = chunk_size

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        raise NotImplementedError