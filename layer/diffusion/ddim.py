"""DDIM sampling layer

DDIM trades exactness for speed by sampling on a reduced set of timesteps, which
often produces similar quality with far fewer denoising steps.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.config.layers.diffusion import DiffusionLayerConfig
from caramba.layer.diffusion import DiffusionDenoiser, DiffusionLayer


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
        timesteps = int(self.config.timesteps)
        steps = int(self.config.infer_steps)
        idx: Tensor = torch.linspace(
            0,
            timesteps - 1,
            steps,
            device=device,
        )

        idx = torch.unique(idx.round().long())

        if idx.numel() < 2:
            return torch.tensor(
                data=[0, timesteps - 1],
                device=device,
                dtype=torch.long,
            )

        return idx