"""Diffusion samplers

Implements DDPM and DDIM sampling with classifier-free guidance (CFG) and
self-conditioning for diffusion-on-embeddings models.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    """Classifier-free guidance configuration

    Controls how conditional and unconditional predictions are mixed during
    sampling.
    """

    guidance_scale: float = 7.5


class DiffusionModelProtocol(Protocol):
    """Interface for diffusion denoisers.

    Concrete implementations should accept the forward signature used here.
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


class _SamplerMixin:
    """Shared sampler utilities (CFG + tokenization).

    Requires:
    - model: DiffusionModelProtocol
    - device: torch.device
    """

    model: DiffusionModelProtocol
    device: torch.device

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
        eps_cond, x0_cond, _ = self.model.forward(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        eps_uncond, x0_uncond, _ = self.model.forward(
            noisy_emb=x,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=None,
            prompt_pad_mask=None,
        )
        scale = float(cfg.guidance_scale)
        guided = eps_uncond + scale * (eps_cond - eps_uncond)
        # x0 is returned only for optional self-conditioning; use the conditional
        # x0 in the guided path by convention.
        _ = x0_uncond
        return guided, x0_cond

    def tokensFromEmbeddings(self, *, emb: Tensor, embedding_weight: Tensor) -> Tensor:
        logits = torch.matmul(emb, embedding_weight.t())
        return logits.argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class DdpmSampler(_SamplerMixin):
    """DDPM sampler

    Uses the full timestep schedule to iteratively denoise from Gaussian noise.
    """

    model: DiffusionModelProtocol
    alpha_bar: Tensor
    timesteps: int
    device: torch.device
    hidden_size: int

    def __post_init__(self) -> None:
        if self.device is None:
            raise ValueError("DdpmSampler.device must not be None")
        if int(self.timesteps) <= 0:
            raise ValueError(f"DdpmSampler.timesteps must be > 0, got {self.timesteps}")
        if int(self.hidden_size) <= 0:
            raise ValueError(f"DdpmSampler.hidden_size must be > 0, got {self.hidden_size}")
        if not isinstance(self.alpha_bar, Tensor):
            raise ValueError("DdpmSampler.alpha_bar must be a torch.Tensor")
        if int(self.alpha_bar.dim()) != 1:
            raise ValueError(
                f"DdpmSampler.alpha_bar must be 1-D, got dim={int(self.alpha_bar.dim())}"
            )
        if int(self.alpha_bar.numel()) != int(self.timesteps):
            raise ValueError(
                "DdpmSampler.alpha_bar length must equal timesteps: "
                f"len={int(self.alpha_bar.numel())} timesteps={int(self.timesteps)}"
            )

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
        for t_index in reversed(range(self.timesteps)):
            x, self_cond = self.ddpmStep(
                x=x,
                t_index=t_index,
                batch_size=batch_size,
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

        return torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)

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

        t = torch.full((batch_size,), t_index, dtype=torch.long, device=self.device)
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
        noise = torch.randn_like(x) if t_index > 0 else torch.zeros_like(x)
        x_next = mean + schedule["betas"][t_index].sqrt() * noise
        return x_next, x0


@dataclass(frozen=True, slots=True)
class DdimSampler(_SamplerMixin):
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

    def __post_init__(self) -> None:
        if self.device is None:
            raise ValueError("DdimSampler.device must not be None")
        if int(self.timesteps) <= 0:
            raise ValueError(f"DdimSampler.timesteps must be > 0, got {self.timesteps}")
        if int(self.hidden_size) <= 0:
            raise ValueError(f"DdimSampler.hidden_size must be > 0, got {self.hidden_size}")
        if int(self.steps) < 2:
            raise ValueError(f"DdimSampler.steps must be >= 2, got {self.steps}")
        if not isinstance(self.alpha_bar, Tensor):
            raise ValueError("DdimSampler.alpha_bar must be a torch.Tensor")
        if int(self.alpha_bar.dim()) != 1:
            raise ValueError(
                f"DdimSampler.alpha_bar must be 1-D, got dim={int(self.alpha_bar.dim())}"
            )
        if int(self.alpha_bar.numel()) != int(self.timesteps):
            raise ValueError(
                "DdimSampler.alpha_bar length must equal timesteps: "
                f"len={int(self.alpha_bar.numel())} timesteps={int(self.timesteps)}"
            )

    def stepIndices(self) -> Tensor:
        idx = torch.linspace(0, int(self.timesteps) - 1, int(self.steps), device=self.device)
        idx = torch.unique(idx.round().long())
        if idx.numel() < 2:
            warnings.warn(
                "DdimSampler.stepIndices produced <2 unique indices; "
                "check __post_init__ validation, self.steps, and self.timesteps "
                f"(steps={int(self.steps)} timesteps={int(self.timesteps)}).",
                RuntimeWarning,
                stacklevel=2,
            )
            if int(self.timesteps) >= 2:
                return torch.tensor([0, int(self.timesteps) - 1], device=self.device, dtype=torch.long)
            if int(self.timesteps) == 1:
                return torch.tensor([0], device=self.device, dtype=torch.long)
            raise ValueError(f"DdimSampler.timesteps must be > 0, got {self.timesteps}")
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
        x = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
        indices = self.stepIndices()
        alpha_bar = self.alpha_bar.to(device=self.device)

        self_cond: Tensor | None = None
        for i in reversed(range(1, int(indices.numel()))):
            x, self_cond = self.ddimStep(
                x=x,
                t=int(indices[i].item()),
                t_prev=int(indices[i - 1].item()),
                batch_size=batch_size,
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
        dir_xt = (1.0 - a_prev - sigma.square()).clamp(min=0.0).sqrt() * eps
        noise = torch.randn_like(x) if float(sigma) > 0.0 else torch.zeros_like(x)
        x_next = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x_next, x0

