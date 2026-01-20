"""Diffusion Model

A diffusion model is a model that denoises a noisy embedding
to generate a clean embedding.
"""
from __future__ import annotations

from torch import Tensor, nn

from config.layers.diffusion import DiffusionLayerConfig


class DiffusionModel(nn.Module):
    """Diffusion Model

    A diffusion model is a model that denoises a noisy embedding
    to generate a clean embedding.
    """
    def __init__(self, config: DiffusionLayerConfig) -> None:
        super().__init__()
        self.config: DiffusionLayerConfig = config

    def forward(  # type: ignore[override]
        self,
        *,
        noisy_emb: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass

        A diffusion denoiser predicts:

        - epsilon (the noise added to the clean embedding)
        - x0 (the predicted clean embedding)
        - optional logits (if the model jointly predicts tokens)

        Caramba samplers (DDPM/DDIM) expect this signature.
        """
        _ = (noisy_emb, t, target_pad_mask, self_cond, prompt_emb, prompt_pad_mask)
        raise NotImplementedError(
            "Subclasses must implement forward pass."
        )