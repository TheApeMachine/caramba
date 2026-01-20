"""Gaussian noise transform

Adds random noise to tensors for data augmentation and regularization,
improving model robustness during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from runtime.tensordict_utils import TensorDictBase, as_tensordict


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    """Add Gaussian noise to tensors

    Injects random noise into data, commonly used for data augmentation or
    regularization during training to improve model robustness.
    """
    src_key: str
    out_key: str
    sigma: float

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Apply noise injection

        Adds zero-mean Gaussian noise scaled by sigma to the source tensor,
        creating a new tensor that preserves the original structure while
        introducing controlled randomness.
        """
        d = dict(td)
        x = d.get(self.src_key, None)
        if not isinstance(x, Tensor):
            return as_tensordict(d)
        noise = torch.randn_like(x) * self.sigma
        d[self.out_key] = x + noise
        return as_tensordict(d)
