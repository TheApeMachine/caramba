"""DDPM sampling layer

This module implements the DDPM reverse process in embedding space, so a model
can generate sequences by repeatedly removing noise instead of predicting tokens
in a single forward pass.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.config.layers.diffusion.ddpm import DDPMLayerConfig
from caramba.layer.diffusion import DiffusionDenoiser, DiffusionLayer


class DdpmSampler(DiffusionLayer):
    """DDPM sampler layer

    This class wraps the DDPM update rule as a composable caramba layer, keeping
    the manifest responsible for hyperparameters while runtime provides the
    conditioning tensors (masks/prompt) for a specific sampling call.
    """
    def __init__(
        self,
        config: DDPMLayerConfig,
        model: DiffusionDenoiser,
        *,
        alpha_bar: Tensor,
    ) -> None:
        super().__init__(config=config, model=model)
        self.alpha_bar: Tensor = alpha_bar

    @torch.no_grad()
    def sampleEmbeddings(
        self,
        *,
        batch_size: int,
        seq_len: int,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
    ) -> Tensor:
        """Sample embeddings with DDPM.

        DDPM uses the full diffusion chain: it iterates from x_T (pure Gaussian
        noise) back to x_0 by repeatedly applying the denoiser
        over all timesteps.
        """
        device = self.alpha_bar.device
        timesteps = int(self.config.timesteps)
        if int(self.alpha_bar.numel()) != timesteps:
            raise ValueError(
                f"alpha_bar length mismatch. Expected numel={timesteps} "
                f"from config.timesteps, got numel={int(self.alpha_bar.numel())}."
            )

        x: Tensor = self.initialNoise(batch_size=batch_size, seq_len=seq_len, device=device)
        schedule = self.makeDdpmSchedule(alpha_bar=self.alpha_bar)

        self_cond: Tensor | None = None
        for t_index in reversed(range(timesteps)):
            x, self_cond = self.ddpmStep(
                x=x,
                t_index=int(t_index),
                batch_size=int(batch_size),
                device=device,
                target_pad_mask=target_pad_mask,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
                schedule=schedule,
                self_cond=self_cond,
            )

        return x

    @torch.no_grad()
    def sampleTokens(
        self,
        *,
        batch_size: int,
        seq_len: int,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        embedding_weight: Tensor,
    ) -> Tensor:
        """Sample token IDs with DDPM.

        This is a convenience wrapper around `sampleEmbeddings` that projects
        the final embeddings back to token IDs using `embedding_weight`.
        """

        emb = self.sampleEmbeddings(
            batch_size=batch_size,
            seq_len=seq_len,
            target_pad_mask=target_pad_mask,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        return self.tokensFromEmbeddings(embedding=emb, embedding_weight=embedding_weight)

    def initialNoise(self, *, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        """Initialize x_T from a standard normal.

        DDPM starts sampling from pure Gaussian noise in the model's embedding
        space. The denoiser then iteratively removes noise.
        """

        return torch.randn(
            int(batch_size),
            int(seq_len),
            int(self.config.hidden_size),
            device=device,
        )

    def makeDdpmSchedule(self, *, alpha_bar: Tensor) -> dict[str, Tensor]:
        """Precompute DDPM schedule tensors.

        The schedule transforms the cumulative product of alphas (alpha_bar)
        into the per-step scalars used in the DDPM reverse update.
        """
        if alpha_bar.dim() != 1:
            raise ValueError(f"Expected alpha_bar to be 1D, got shape={tuple(alpha_bar.shape)}")

        device = alpha_bar.device
        alpha_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]])
        alphas = alpha_bar / alpha_prev
        betas = 1.0 - alphas
        return {
            "betas": betas,
            "sqrt_recip_alpha": (1.0 / alphas).sqrt(),
            "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
        }

    def ddpmStep(
        self,
        *,
        x: Tensor,
        t_index: int,
        batch_size: int,
        device: torch.device,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        schedule: dict[str, Tensor],
        self_cond: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """One reverse diffusion step for DDPM."""
        t = torch.full((int(batch_size),), int(t_index), dtype=torch.long, device=device)
        eps, x0 = self.predictEpsAndX0(
            x=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        mean = schedule["sqrt_recip_alpha"][t_index] * (
            x - schedule["betas"][t_index] / schedule["sqrt_one_minus_alpha_bar"][t_index] * eps
        )
        x_next = mean + (schedule["betas"][t_index].sqrt() * torch.randn_like(x) if t_index > 0 else 0.0)
        return x_next, x0
