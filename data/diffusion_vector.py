"""Diffusion vector dataset

Generates training samples for diffusion models by adding noise to data vectors
at random timesteps. This implements the core diffusion training objective
(predict noise) without requiring external diffusion libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict

def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    """Linear noise schedule

    Creates a linear schedule for the noise level (beta) across diffusion
    timesteps, which controls how much noise is added at each step. Linear
    schedules are simple and work well for many tasks.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


@dataclass(frozen=True, slots=True)
class DiffusionVectorDataset:
    """Diffusion vector dataset component

    Manifest-level dataset that loads vector data and generates (noisy_data,
    timestep, noise) triplets for diffusion model training. Each sample
    randomly selects a timestep and adds the corresponding amount of noise.
    """

    x_path: str
    timesteps: int = 1000

    def build(self) -> Dataset[TensorDictBase]:
        """Build diffusion dataset

        Loads the clean data, computes the noise schedule, and creates a
        dataset that samples random timesteps and generates noisy versions
        of the data for training the denoising model.
        """
        x0 = np.load(Path(self.x_path))
        x0_t = torch.from_numpy(x0).float()
        betas = _linear_beta_schedule(int(self.timesteps))
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)

        class _DS(Dataset[TensorDictBase]):
            def __len__(self) -> int:
                return int(x0_t.shape[0])

            def __getitem__(self, idx: int) -> TensorDictBase:
                x0_i = x0_t[idx]
                t = torch.randint(low=0, high=alpha_bar.shape[0], size=(1,), dtype=torch.long)[0]
                eps = torch.randn_like(x0_i)
                ab = alpha_bar[t].to(dtype=x0_i.dtype)
                x_t = torch.sqrt(ab) * x0_i + torch.sqrt(1.0 - ab) * eps
                return as_tensordict({"x_t": x_t, "t": t, "eps": eps})

        return _DS()

