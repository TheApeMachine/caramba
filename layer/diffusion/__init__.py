"""Diffusion layer primitives

Diffusion layers generate in embedding space by iteratively denoising noise into
structure, which is a useful alternative to sampling tokens directly from a
single logits distribution.
"""
from __future__ import annotations

import math
from typing import Protocol
import torch
from torch import Tensor, nn

from config.layers.diffusion import DiffusionLayerConfig


class DiffusionDenoiser(Protocol):
    """Diffusion denoiser protocol

    This protocol defines the sampler-facing API, so different denoiser models
    can plug into the same sampler logic without coupling sampling to one model
    implementation.
    """
    def forward(
        self,
        *,
        noisy_emb: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


class DiffusionLayer(nn.Module):
    """Diffusion layer base class

    This base class holds shared utilities (CFG mixing, projection back to
    tokens) so individual samplers can focus on the schedule math.
    """
    def __init__(
        self,
        config: DiffusionLayerConfig,
        model: DiffusionDenoiser,
    ) -> None:
        super().__init__()
        self.config: DiffusionLayerConfig = config
        self.model: DiffusionDenoiser = model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Diffusion layers typically expose explicit sampling methods; `forward`
        is left abstract so a topology can decide how to invoke sampling.
        """
        raise NotImplementedError(
            "Subclasses must implement forward pass."
        )

    def tokensFromEmbeddings(
        self,
        *,
        embedding: Tensor,
        embedding_weight: Tensor,
    ) -> Tensor:
        """Convert embeddings to token IDs

        Diffusion samplers naturally operate in embedding space. Converting back
        to discrete token IDs is done by scoring each position against an
        embedding matrix and selecting the argmax token ID.
        """
        return torch.matmul(
            input=embedding,
            other=embedding_weight.t(),
        ).argmax(dim=-1)

    def predictEpsAndX0(
        self,
        *,
        x: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        guidance_scale: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict epsilon and x0 with classifier-free guidance

        The denoiser predicts the noise component (epsilon) and the clean
        embedding (x0). Classifier-free guidance (CFG) mixes conditional and
        unconditional predictions to steer sampling toward the prompt.
        """
        scale = float(
            self.config.cfg_guidance_scale
        ) if guidance_scale is None else float(guidance_scale)

        if math.isclose(scale, 1.0):
            eps_cond, x0_cond, _ = self.model.forward(
                noisy_emb=x,
                t=t,
                target_pad_mask=target_pad_mask,
                self_cond=self_cond,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
            )
            return eps_cond, x0_cond

        if math.isclose(scale, 0.0):
            eps_uncond, x0_uncond, _ = self.model.forward(
                noisy_emb=x,
                t=t,
                target_pad_mask=target_pad_mask,
                self_cond=self_cond,
                prompt_emb=None,
                prompt_pad_mask=None,
            )
            return eps_uncond, x0_uncond

        eps_cond, x0_cond, _ = self.model.forward(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        eps_uncond, _, _ = self.model.forward(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=None,
            prompt_pad_mask=None,
        )
        guided = eps_uncond + scale * (eps_cond - eps_uncond)
        return guided, x0_cond

    # Note: `tokensFromEmbeddings` is the single supported API; no aliases.
