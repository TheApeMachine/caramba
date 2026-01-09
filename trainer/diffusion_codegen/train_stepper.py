"""Training step logic for diffusion codegen

Computes diffusion losses and performs gradient accumulation steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as torchF
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from caramba.config.train import TrainConfig
from caramba.runtime.tensordict_utils import to_device
from caramba.trainer.diffusion_codegen.train_context import TrainingContext


@dataclass(frozen=True, slots=True)
class StepSettings:
    """Loss/conditioning settings."""

    timesteps: int
    mse_lambda: float
    unconditional_prob: float
    self_condition_prob: float
    grad_clip: float


@dataclass(frozen=True, slots=True)
class TrainStepper:
    """Compute and apply a training step."""

    settings: StepSettings

    def step(
        self,
        *,
        batch: Any,
        ctx: TrainingContext,
        train: TrainConfig,
        optimizer: Optimizer,
        scheduler: CosineAnnealingLR | None,
    ) -> float:
        batch = to_device(batch, device=ctx.device)
        loss = self.backward(batch=batch, ctx=ctx, train=train)
        if self.shouldStepOptimizer(ctx=ctx, train=train):
            self.stepOptimizer(model=ctx.model, ctx=ctx, optimizer=optimizer, scheduler=scheduler, clip=float(self.settings.grad_clip))
        return float(loss)

    def backward(self, *, batch: Any, ctx: TrainingContext, train: TrainConfig) -> Tensor:
        target_ids: Tensor = batch["target_ids"]
        prompt_ids: Tensor = batch["prompt_ids"]
        target_pad = target_ids.eq(int(ctx.pad_id))
        prompt_pad = prompt_ids.eq(int(ctx.pad_id))

        clean = ctx.model.embedding(target_ids)
        prompt = ctx.model.embedding(prompt_ids)
        t = torch.randint(low=0, high=int(self.settings.timesteps), size=(int(target_ids.shape[0]),), device=ctx.device, dtype=torch.long)
        noise = torch.randn_like(clean)
        noisy = self.noisyEmbeddings(ctx=ctx, t=t, clean=clean, noise=noise)

        unconditional = torch.rand((), device=ctx.device) < float(self.settings.unconditional_prob)
        prompt_emb = None if bool(unconditional) else prompt
        prompt_mask = None if bool(unconditional) else prompt_pad

        loss = self.loss(
            ctx=ctx,
            noisy=noisy,
            t=t,
            target_pad=target_pad,
            prompt_emb=prompt_emb,
            prompt_pad=prompt_mask,
            target_ids=target_ids,
            noise=noise,
            use_amp=self.useAmp(train=train, device=ctx.device),
            grad_accum=self.gradAccum(train=train),
        )
        ctx.scaler.scale(loss).backward()
        ctx.step_counter += 1
        return loss.detach()

    def loss(
        self,
        *,
        ctx: TrainingContext,
        noisy: Tensor,
        t: Tensor,
        target_pad: Tensor,
        prompt_emb: Tensor | None,
        prompt_pad: Tensor | None,
        target_ids: Tensor,
        noise: Tensor,
        use_amp: bool,
        grad_accum: int,
    ) -> Tensor:
        mse_weight = float(self.settings.mse_lambda)
        use_sc = torch.rand((), device=noisy.device) < float(self.settings.self_condition_prob)
        with torch.autocast(device_type="cuda", enabled=bool(use_amp)):
            eps1, x0_1, logits1 = ctx.model(
                noisy_emb=noisy,
                t=t,
                target_pad_mask=target_pad,
                self_cond=None,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad,
            )
            eps, logits = (eps1, logits1) if not bool(use_sc) else self.selfConditioned(ctx=ctx, noisy=noisy, t=t, target_pad=target_pad, x0=x0_1.detach(), prompt_emb=prompt_emb, prompt_pad=prompt_pad)
            loss_mse = torchF.mse_loss(eps, noise)
            loss_ce = torchF.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=int(getattr(ctx.model, "pad_id", 0)))
            return (mse_weight * loss_mse + (1.0 - mse_weight) * loss_ce) / max(1, int(grad_accum))

    def selfConditioned(
        self,
        *,
        ctx: TrainingContext,
        noisy: Tensor,
        t: Tensor,
        target_pad: Tensor,
        x0: Tensor,
        prompt_emb: Tensor | None,
        prompt_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        eps2, _x0_2, logits2 = ctx.model(
            noisy_emb=noisy,
            t=t,
            target_pad_mask=target_pad,
            self_cond=x0,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad,
        )
        return eps2, logits2

    def noisyEmbeddings(self, *, ctx: TrainingContext, t: Tensor, clean: Tensor, noise: Tensor) -> Tensor:
        ab = ctx.schedule.gatherToShape(values=ctx.alpha_bar, t=t, shape=clean.shape)
        return ab.sqrt() * clean + (1.0 - ab).sqrt() * noise

    def stepOptimizer(
        self,
        *,
        model: Any,
        ctx: TrainingContext,
        optimizer: Optimizer,
        scheduler: CosineAnnealingLR | None,
        clip: float,
    ) -> None:
        if float(clip) > 0.0:
            ctx.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip))
        ctx.scaler.step(optimizer)
        ctx.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    def shouldStepOptimizer(self, *, ctx: TrainingContext, train: TrainConfig) -> bool:
        return int(ctx.step_counter + 1) % max(1, self.gradAccum(train=train)) == 0

    def gradAccum(self, *, train: TrainConfig) -> int:
        return int(getattr(train, "gradient_accumulation_steps", 1))

    def useAmp(self, *, train: TrainConfig, device: torch.device) -> bool:
        return bool(getattr(train, "use_amp", False)) and device.type == "cuda"

