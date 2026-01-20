"""DDIM sampling layer

DDIM trades exactness for speed by sampling on a reduced set of timesteps, which
often produces similar quality with far fewer denoising steps.
"""
from __future__ import annotations

import torch
from torch import Tensor

from config.layers.diffusion import DiffusionLayerConfig
from layer.diffusion import DiffusionDenoiser, DiffusionLayer


class DdimSampler(DiffusionLayer):
    """DDIM sampler layer

    The key idea is that you do not need to traverse every diffusion step to get
    a usable sample; picking a coarser schedule can make sampling practical at
    inference time.
    """
    def __init__(
        self,
        config: DiffusionLayerConfig,
        model: DiffusionDenoiser,
    ) -> None:
        super().__init__(config=config, model=model)

    def step_indices(self, *, device: torch.device) -> Tensor:
        """Compute DDIM step indices

        DDIM chooses a subset of the training timesteps; rounding and uniquing
        keep indices valid and monotonic even for small schedules.
        """
        if self.config.timesteps is None:
            raise ValueError("DDIM requires config.timesteps to be set (got None)")
        if self.config.infer_steps is None:
            raise ValueError("DDIM requires config.infer_steps to be set (got None)")
        try:
            timesteps = int(self.config.timesteps)
        except Exception as e:
            raise ValueError(f"DDIM config.timesteps must be castable to int, got {self.config.timesteps!r}") from e
        try:
            steps = int(self.config.infer_steps)
        except Exception as e:
            raise ValueError(f"DDIM config.infer_steps must be castable to int, got {self.config.infer_steps!r}") from e
        if timesteps <= 0:
            raise ValueError(f"DDIM requires config.timesteps > 0, got {timesteps}")
        if steps <= 0:
            raise ValueError(f"DDIM requires config.infer_steps > 0, got {steps}")

        idx: Tensor = torch.linspace(
            0,
            timesteps - 1,
            steps,
            device=device,
        )

        idx = torch.unique(idx.round().long())

        if idx.numel() < 2:
            if timesteps >= 2:
                return torch.tensor(
                    data=[0, timesteps - 1],
                    device=device,
                    dtype=torch.long,
                )
            if timesteps == 1:
                return torch.tensor(
                    data=[0],
                    device=device,
                    dtype=torch.long,
                )
            raise ValueError(f"DDIM requires timesteps > 0, got {timesteps}")

        return idx