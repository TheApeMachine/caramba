"""MOSAIC forward paths

Contains the two compute paths for MOSAIC blocks:
- FastTrainPath: chunked training path (T>1) to reduce overhead
- SequentialPath: exact streaming semantics path
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from torch import Tensor, nn

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.state_bank import StateBank
from caramba.layer.mosaic.memory import MosaicMemory


class Path(nn.Module, ABC):
    """MOSAIC forward path

    Base class for MOSAIC forward-path implementations.

    Subclassing notes
    ---------------
    - `forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor` must be implemented by
      subclasses. It is expected to consume the MOSAIC block inputs (typically token
      activations and routing/control tensors) and return an output tensor of shape
      `(B, T, D)` on the same device/dtype as the inputs.
    - Implementations may maintain state through `self.state` / `self.memory` and should
      avoid mutating input tensors in-place unless explicitly intended.

    Parameters
    ----------
    state:
        The state bank backing the block (scan/step state updates).
    memory:
        The MOSAIC memory module (read/write + optional RMF updates).
    gate_long, gate_mem:
        Learned gates used to combine local, long-state, and memory readouts.
    chunk_size:
        Execution chunk size for vectorized paths. `SequentialPath` uses 1; `FastTrainPath`
        uses a larger chunk size for speed.

    Provided variants
    -----------------
    - `FastTrainPath`: chunked execution optimized for training (lower Python overhead).
    - `SequentialPath`: exact token-by-token semantics, used for decoding and as a
      correctness baseline.
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

    @abstractmethod
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Run the path forward pass (subclasses must implement)."""