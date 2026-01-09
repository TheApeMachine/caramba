"""Rename keys transform

Changes dictionary key names in TensorDicts, useful for adapting datasets
with different naming conventions to match model expectations.
"""
from __future__ import annotations

from dataclasses import dataclass

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


@dataclass(frozen=True, slots=True)
class RenameKeys:
    """Rename dictionary keys

    Changes key names in a TensorDict, useful for adapting datasets that use
    different naming conventions to match what your model expects.
    """
    mapping: dict[str, str]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Apply key renaming

        Creates a new TensorDict with renamed keys, preserving all values while
        updating the key names according to the mapping.
        """
        d = dict(td)
        for src, dst in self.mapping.items():
            if src in d:
                d[dst] = d.pop(src)
        return as_tensordict(d)