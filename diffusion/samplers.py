"""Diffusion samplers

Implements DDPM and DDIM sampling with classifier-free guidance (CFG) and
self-conditioning for diffusion-on-embeddings models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    """Classifier-free guidance configuration

    Controls how conditional and unconditional predictions are mixed during
    sampling.
    """

    guidance_scale: float = 7.5


class DiffusionModelProtocol(nn.Module):
    """Protocol-like base for diffusion denoisers.

    Concrete implementations should accept the forward signature used here.
    """

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
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DdpmSampler:
    """DDPM sampler

    Uses the full timestep schedule to iteratively denoise from Gaussian noise.
    """

    model: DiffusionModelProtocol
    alpha_bar: Tensor
    timesteps: int
    device: torch.device
    hidden_size: int

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        seq_len: int,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
        embedding_weight: Tensor,
    ) -> Tensor:
        x = self.initialNoise(batch_size=batch_size, seq_len=seq_len)
        schedule = self.makeDdpmSchedule()

        self_cond: Tensor | None = None
        for t_index in reversed(range(int(self.timesteps))):
            x, self_cond = self.ddpmStep(
                x=x,
                t_index=int(t_index),
                batch_size=int(batch_size),
                target_pad_mask=target_pad_mask,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
                cfg=cfg,
                schedule=schedule,
                self_cond=self_cond,
            )

        return self.tokensFromEmbeddings(emb=x, embedding_weight=embedding_weight)

    def initialNoise(self, *, batch_size: int, seq_len: int) -> Tensor:
        """Initialize x_T from standard normal."""

        return torch.randn(int(batch_size), int(seq_len), int(self.hidden_size), device=self.device)

    def makeDdpmSchedule(self) -> dict[str, Tensor]:
        """Precompute DDPM schedule tensors."""

        alpha_bar = self.alpha_bar.to(device=self.device)
        alpha_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bar[:-1]])
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
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
        schedule: dict[str, Tensor],
        self_cond: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """One reverse diffusion step for DDPM."""

        t = torch.full((int(batch_size),), int(t_index), dtype=torch.long, device=self.device)
        eps, x0 = self.predictEpsAndX0(
            x=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
            cfg=cfg,
        )
        mean = schedule["sqrt_recip_alpha"][t_index] * (
            x - schedule["betas"][t_index] / schedule["sqrt_one_minus_alpha_bar"][t_index] * eps
        )
        x_next = mean + (schedule["betas"][t_index].sqrt() * torch.randn_like(x) if t_index > 0 else 0.0)
        return x_next, x0

    def predictEpsAndX0(
        self,
        *,
        x: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
    ) -> tuple[Tensor, Tensor]:
        eps_cond, x0_cond, _ = self.model(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        eps_uncond, _, _ = self.model(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=None,
            prompt_pad_mask=None,
        )
        guided = eps_uncond + float(cfg.guidance_scale) * (eps_cond - eps_uncond)
        return guided, x0_cond

    def tokensFromEmbeddings(self, *, emb: Tensor, embedding_weight: Tensor) -> Tensor:
        logits = torch.matmul(emb, embedding_weight.t())
        return logits.argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class DdimSampler:
    """DDIM sampler

    Uses a reduced set of timesteps for faster sampling, with optional eta noise.
    """

    model: DiffusionModelProtocol
    alpha_bar: Tensor
    timesteps: int
    device: torch.device
    hidden_size: int
    steps: int = 50
    eta: float = 0.0

    def stepIndices(self) -> Tensor:
        idx = torch.linspace(0, int(self.timesteps) - 1, int(self.steps), device=self.device)
        idx = torch.unique(idx.round().long())
        if idx.numel() < 2:
            return torch.tensor([0, int(self.timesteps) - 1], device=self.device, dtype=torch.long)
        return idx

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        seq_len: int,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
        embedding_weight: Tensor,
    ) -> Tensor:
        x = torch.randn(int(batch_size), int(seq_len), int(self.hidden_size), device=self.device)
        indices = self.stepIndices()
        alpha_bar = self.alpha_bar.to(device=self.device)

        self_cond: Tensor | None = None
        for i in reversed(range(1, int(indices.numel()))):
            x, self_cond = self.ddimStep(
                x=x,
                t=int(indices[i].item()),
                t_prev=int(indices[i - 1].item()),
                batch_size=int(batch_size),
                alpha_bar=alpha_bar,
                target_pad_mask=target_pad_mask,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
                cfg=cfg,
                self_cond=self_cond,
            )

        return self.tokensFromEmbeddings(emb=x, embedding_weight=embedding_weight)

    def ddimStep(
        self,
        *,
        x: Tensor,
        t: int,
        t_prev: int,
        batch_size: int,
        alpha_bar: Tensor,
        target_pad_mask: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
        self_cond: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """One reverse diffusion step for DDIM."""

        a_t = alpha_bar[int(t)]
        a_prev = alpha_bar[int(t_prev)]
        t_vec = torch.full((int(batch_size),), int(t), dtype=torch.long, device=self.device)

        eps, x0 = self.predictEpsAndX0(
            x=x,
            t=t_vec,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
            cfg=cfg,
        )

        x0_pred = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = float(self.eta) * ((1.0 - a_prev) / (1.0 - a_t)).sqrt() * (1.0 - a_t / a_prev).sqrt()
        dir_xt = (1.0 - a_prev - sigma**2).sqrt() * eps
        noise = torch.randn_like(x) if sigma > 0.0 else 0.0
        x_next = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x_next, x0

    def predictEpsAndX0(
        self,
        *,
        x: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
        cfg: GuidanceConfig,
    ) -> tuple[Tensor, Tensor]:
        eps_cond, x0_cond, _ = self.model(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        eps_uncond, _, _ = self.model(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=None,
            prompt_pad_mask=None,
        )
        guided = eps_uncond + float(cfg.guidance_scale) * (eps_cond - eps_uncond)
        return guided, x0_cond

    def tokensFromEmbeddings(self, *, emb: Tensor, embedding_weight: Tensor) -> Tensor:
        logits = torch.matmul(emb, embedding_weight.t())
        return logits.argmax(dim=-1)

