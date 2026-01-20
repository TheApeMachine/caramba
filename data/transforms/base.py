"""Transform base class

Transforms are recursively composable, so each transfor can be an initialize
value for any other transform.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from runtime.tensordict_utils import TensorDictBase
from data.transforms.compose import Compose


class Transform(ABC):
    """Transform protocol

    A transform is a function that takes a TensorDict and returns a TensorDict.
    """
    def __init__(self, pipeline: Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        for t in self.pipeline.transforms:
            out: TensorDictBase = t(td)

        return out
