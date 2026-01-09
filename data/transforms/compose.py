"""Transform composition

Defines the transform protocol and composition class for chaining multiple
transforms into pipelines, enabling complex preprocessing from simple pieces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from caramba.runtime.tensordict_utils import TensorDictBase


class Transform(Protocol):
    """Transform protocol

    Defines the interface that all transforms must follow: take a TensorDict
    in, return a TensorDict out. This protocol enables type checking and
    makes it easy to compose transforms into pipelines.
    """
    def __call__(self, td: TensorDictBase) -> TensorDictBase: ...


@dataclass(frozen=True, slots=True)
class Compose:
    """Transform composition

    Chains multiple transforms together into a pipeline, applying them
    sequentially so complex preprocessing can be built from simple pieces.
    """
    transforms: list[Transform]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Apply transform pipeline

        Runs each transform in sequence, passing the output of one as input
        to the next, enabling complex data transformations through composition.
        """
        out = td
        for t in self.transforms:
            out = t(out)
        return out
