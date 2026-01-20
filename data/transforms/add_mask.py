"""Add mask transform

Creates attention masks from ignore indices, enabling models to distinguish
between valid tokens and padding without manual masking logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from runtime.tensordict_utils import TensorDictBase, as_tensordict

@dataclass(frozen=True, slots=True)
class AddMask:
    """Add attention mask from ignore index

    Creates a boolean mask indicating which positions are valid (not padding),
    which attention mechanisms use to ignore padding tokens during computation.
    """
    src_key: str
    mask_key: str
    ignore_index: int

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Generate mask tensor

        Produces a mask where True indicates valid positions and False marks
        padding tokens, enabling models to ignore padding without manual
        masking logic.
        """
        d = dict(td)
        src = d.get(self.src_key, None)
        if not isinstance(src, Tensor):
            return as_tensordict(d)
        d[self.mask_key] = (src != self.ignore_index)
        return as_tensordict(d)
