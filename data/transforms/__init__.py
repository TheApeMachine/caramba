"""Data transforms package

Small, composable functions that modify TensorDicts between dataset loading
and model input. Keeping transforms separate from datasets makes it easy to
experiment with different preprocessing pipelines without changing data code.
"""
from __future__ import annotations

from caramba.data.transforms.add_mask import AddMask
from caramba.data.transforms.cast_dtype import CastDtype
from caramba.data.transforms.compose import Compose
from caramba.data.transforms.gaussian_noise import GaussianNoise
from caramba.data.transforms.graph_batch import GraphBatch
from caramba.data.transforms.rename_keys import RenameKeys
from caramba.data.transforms.token_shift import TokenShift
from caramba.data.transforms.base import Transform


__all__ = [
    "AddMask",
    "CastDtype",
    "Compose",
    "GaussianNoise",
    "GraphBatch",
    "RenameKeys",
    "TokenShift",
    "Transform",
]
