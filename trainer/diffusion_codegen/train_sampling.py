"""Sampling during training for diffusion codegen."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from caramba.console import logger
from caramba.diffusion.samplers import DdimSampler, DdpmSampler, GuidanceConfig
from caramba.trainer.diffusion_codegen.ema import ExponentialMovingAverage
from caramba.trainer.diffusion_codegen.prompt import PromptEncoder
from caramba.trainer.diffusion_codegen.train_context import TrainingContext


@dataclass(frozen=True, slots=True)
class SamplingDuringTraining:
    """Periodic sampler used by the training loop."""

    sampler_config: dict[str, Any] | None
    timesteps: int

    def maybeSample(self, *, checkpoint_dir: Path, ctx: TrainingContext, step: int, ema: ExponentialMovingAverage | None) -> Path | None:
        cfg = self.sampler_config or {}
        every = int(cfg.get("every", 0))
        if every <= 0 or int(step) % int(every) != 0:
            return None
        num_samples = int(cfg.get("num_samples", 1))
        seq_len = int(cfg.get("seq_len", int(ctx.seq_len)))
        return self.writeSamples(checkpoint_dir=checkpoint_dir, ctx=ctx, step=step, seq_len=seq_len, num_samples=num_samples, ema=ema)

    def writeSamples(
        self,
        *,
        checkpoint_dir: Path,
        ctx: TrainingContext,
        step: int,
        seq_len: int,
        num_samples: int,
        ema: ExponentialMovingAverage | None,
    ) -> Path:
        if ema:
            ema.apply(model=ctx.model)
        try:
            tokens = self.sampleTokens(ctx=ctx, seq_len=int(seq_len), num_samples=int(num_samples))
            texts = [ctx.tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
            out = checkpoint_dir / f"samples_step{int(step):08d}_{int(time.time())}.txt"
            out.write_text("\\n\\n".join(texts), encoding="utf-8")
            logger.path(str(out), label="samples")
            return out
        finally:
            if ema:
                ema.restore(model=ctx.model)

    def sampleTokens(self, *, ctx: TrainingContext, seq_len: int, num_samples: int) -> Tensor:
        cfg = self.sampler_config or {}
        kind = str(cfg.get("kind", "ddim")).lower().strip()
        guidance = GuidanceConfig(guidance_scale=float(cfg.get("guidance_scale", 7.5)))
        prompt_emb, prompt_pad = self.promptForSampling(ctx=ctx, seq_len=seq_len)
        if kind == "ddim":
            return self.sampleDdim(ctx=ctx, seq_len=seq_len, num_samples=num_samples, prompt_emb=prompt_emb, prompt_pad=prompt_pad, guidance=guidance, cfg=cfg)
        if kind == "ddpm":
            return self.sampleDdpm(ctx=ctx, seq_len=seq_len, num_samples=num_samples, prompt_emb=prompt_emb, prompt_pad=prompt_pad, guidance=guidance)
        raise ValueError(f"Unknown sampler kind: {kind!r}")

    def sampleDdim(
        self,
        *,
        ctx: TrainingContext,
        seq_len: int,
        num_samples: int,
        prompt_emb: Tensor | None,
        prompt_pad: Tensor | None,
        guidance: GuidanceConfig,
        cfg: dict[str, Any],
    ) -> Tensor:
        sampler = DdimSampler(
            model=ctx.model,
            alpha_bar=ctx.alpha_bar,
            timesteps=int(self.timesteps),
            device=ctx.device,
            hidden_size=int(ctx.hidden_size),
            steps=int(cfg.get("ddim_steps", 50)),
            eta=float(cfg.get("ddim_eta", 0.0)),
        )
        return sampler.sample(
            batch_size=int(num_samples),
            seq_len=int(seq_len),
            target_pad_mask=None,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad,
            cfg=guidance,
            embedding_weight=ctx.model.embedding.weight,
        )

    def sampleDdpm(
        self,
        *,
        ctx: TrainingContext,
        seq_len: int,
        num_samples: int,
        prompt_emb: Tensor | None,
        prompt_pad: Tensor | None,
        guidance: GuidanceConfig,
    ) -> Tensor:
        sampler = DdpmSampler(
            model=ctx.model,
            alpha_bar=ctx.alpha_bar,
            timesteps=int(self.timesteps),
            device=ctx.device,
            hidden_size=int(ctx.hidden_size),
        )
        return sampler.sample(
            batch_size=int(num_samples),
            seq_len=int(seq_len),
            target_pad_mask=None,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad,
            cfg=guidance,
            embedding_weight=ctx.model.embedding.weight,
        )

    def promptForSampling(self, *, ctx: TrainingContext, seq_len: int) -> tuple[Tensor | None, Tensor | None]:
        cfg = self.sampler_config or {}
        prompt = cfg.get("prompt", None)
        if not isinstance(prompt, str) or not prompt:
            return None, None
        encoder = PromptEncoder(tokenizer=ctx.tokenizer, embedding=ctx.model.embedding, pad_id=int(ctx.pad_id), seq_len=int(seq_len))
        return encoder.encode(prompt=str(prompt), device=ctx.device)

