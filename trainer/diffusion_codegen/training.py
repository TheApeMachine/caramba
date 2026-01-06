"""Training orchestration for diffusion codegen.

This module keeps the high-level target/run orchestration small and delegates
the step logic and sampling logic to dedicated helpers to keep files and
methods within Caramba's size limits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.console import logger
from caramba.diffusion.schedule import NoiseSchedule
from caramba.trainer.diffusion_codegen.checkpoints import CheckpointManager
from caramba.trainer.diffusion_codegen.ema import ExponentialMovingAverage
from caramba.trainer.diffusion_codegen.optim import OptimizerFactory
from caramba.trainer.diffusion_codegen.train_context import TrainingContext
from caramba.trainer.diffusion_codegen.train_loader import LoaderFactory
from caramba.trainer.diffusion_codegen.train_sampling import SamplingDuringTraining
from caramba.trainer.diffusion_codegen.train_stepper import StepSettings, TrainStepper


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Settings that shape diffusion-codegen training."""

    timesteps: int
    schedule: str
    beta_start: float
    beta_end: float
    mse_lambda: float
    unconditional_prob: float
    self_condition_prob: float
    grad_clip: float
    use_ema: bool
    ema_decay: float
    sampler: dict[str, Any] | None


@dataclass
class TrainingRunner:
    """Train diffusion-codegen targets."""

    settings: TrainingSettings
    optimizerFactory: OptimizerFactory
    loaderFactory: LoaderFactory

    def runTarget(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        dataset_comp: object,
        system: object,
        checkpoint_dir: Path,
        tokenizer_file: str,
    ) -> None:
        runs = [r for r in target.runs if r.train is not None]
        if not runs:
            raise ValueError("Train action requires at least one run with train config.")
        for run in runs:
            seed = int(run.seed) if isinstance(run.seed, int) else int(run.seed[0])
            train = self.requireTrainConfig(train=run.train)
            self.runSingle(manifest=manifest, target=target, run_id=str(run.id), steps=int(run.steps), seed=seed, train=train, dataset_comp=dataset_comp, system=system, checkpoint_dir=checkpoint_dir, tokenizer_file=tokenizer_file)

    def runSingle(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        run_id: str,
        steps: int,
        seed: int,
        train: TrainConfig,
        dataset_comp: object,
        system: object,
        checkpoint_dir: Path,
        tokenizer_file: str,
    ) -> None:
        torch.manual_seed(int(seed))
        device = torch.device(str(train.device))
        system = self.moveSystem(system=system, device=device, dtype=self.resolveDtype(dtype=str(train.dtype)))
        model = self.unwrapModel(system=system)
        loader = self.loaderFactory.build(dataset_comp=dataset_comp, train=train, device=device)

        schedule = NoiseSchedule(kind=str(self.settings.schedule), beta_start=float(self.settings.beta_start), beta_end=float(self.settings.beta_end))
        alpha_bar = schedule.alphasCumprod(timesteps=int(self.settings.timesteps), device=device)
        ctx = TrainingContext.fromTokenizerFile(model=model, tokenizer_file=tokenizer_file, alpha_bar=alpha_bar, schedule=schedule)

        optimizer = self.optimizerFactory.buildOptimizer(model=model, train=train)
        scheduler = self.optimizerFactory.buildScheduler(optimizer=optimizer, train=train, steps=int(steps))
        ema = self.buildEma(model=model) if bool(self.settings.use_ema) else None
        stepper = TrainStepper(settings=self.stepSettings())
        sampler = SamplingDuringTraining(sampler_config=self.settings.sampler, timesteps=int(self.settings.timesteps))
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        save_every = int(getattr(getattr(manifest.defaults, "runtime", object()), "save_every", 200))

        self.loop(steps=steps, run_id=run_id, target=target, train=train, loader=loader, ctx=ctx, stepper=stepper, sampler=sampler, ema=ema, optimizer=optimizer, scheduler=scheduler, manager=manager, checkpoint_dir=checkpoint_dir, tokenizer_file=tokenizer_file, save_every=save_every)

    def loop(
        self,
        *,
        steps: int,
        run_id: str,
        target: ExperimentTargetConfig,
        train: TrainConfig,
        loader: object,
        ctx: TrainingContext,
        stepper: TrainStepper,
        sampler: SamplingDuringTraining,
        ema: ExponentialMovingAverage | None,
        optimizer: object,
        scheduler: object,
        manager: CheckpointManager,
        checkpoint_dir: Path,
        tokenizer_file: str,
        save_every: int,
    ) -> None:
        tic = time.time()
        for step_index, batch in enumerate(loader, start=1):  # type: ignore[arg-type]
            if int(step_index) > int(steps):
                break
            loss = stepper.step(batch=batch, ctx=ctx, train=train, optimizer=optimizer, scheduler=scheduler)  # type: ignore[arg-type]
            if int(step_index) % 10 == 0:
                logger.info(f"step={step_index} loss={loss:.4f} dt={time.time()-tic:.2f}s")
                tic = time.time()
            if ema and stepper.shouldStepOptimizer(ctx=ctx, train=train):
                ema.update(model=ctx.model)
            if int(step_index) % int(save_every) == 0 or int(step_index) == int(steps):
                self.saveCheckpoint(manager=manager, run_id=run_id, step=step_index, ctx=ctx, optimizer=optimizer, scheduler=scheduler, ema=ema, target=target, train=train, tokenizer_file=tokenizer_file)
                sampler.maybeSample(checkpoint_dir=checkpoint_dir, ctx=ctx, step=int(step_index), ema=ema)

    def saveCheckpoint(
        self,
        *,
        manager: CheckpointManager,
        run_id: str,
        step: int,
        ctx: TrainingContext,
        optimizer: object,
        scheduler: object,
        ema: ExponentialMovingAverage | None,
        target: ExperimentTargetConfig,
        train: TrainConfig,
        tokenizer_file: str,
    ) -> None:
        payload = self.checkpointPayload(target=target, train=train, tokenizer_file=tokenizer_file)
        out = manager.save(
            run_id=str(run_id),
            step=int(step),
            model_state=ctx.model.state_dict(),
            optimizer_state=optimizer.state_dict(),  # type: ignore[attr-defined]
            scheduler_state=scheduler.state_dict() if scheduler else None,  # type: ignore[attr-defined]
            ema_state=ema.state_dict() if ema else None,
            payload=payload,
        )
        logger.path(str(out), label="checkpoint")

    def stepSettings(self) -> StepSettings:
        return StepSettings(
            timesteps=int(self.settings.timesteps),
            mse_lambda=float(self.settings.mse_lambda),
            unconditional_prob=float(self.settings.unconditional_prob),
            self_condition_prob=float(self.settings.self_condition_prob),
            grad_clip=float(self.settings.grad_clip),
        )

    def buildEma(self, *, model: nn.Module) -> ExponentialMovingAverage:
        ema = ExponentialMovingAverage(decay=float(self.settings.ema_decay))
        ema.register(model=model)
        return ema

    def requireTrainConfig(self, *, train: TrainConfig | None) -> TrainConfig:
        if train is None:
            raise ValueError("Expected run.train to be present for train action.")
        if train.phase != TrainPhase.STANDARD:
            raise ValueError(f"diffusion_codegen requires train.phase=standard, got {train.phase}")
        return train

    def resolveDtype(self, *, dtype: str) -> torch.dtype:
        d = str(dtype).lower().strip()
        if d in {"float32", "fp32"}:
            return torch.float32
        if d in {"float16", "fp16"}:
            return torch.float16
        if d in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype!r}")

    def moveSystem(self, *, system: object, device: torch.device, dtype: torch.dtype) -> object:
        if not hasattr(system, "to"):
            raise TypeError("System component does not expose to(device=..., dtype=...)")
        return system.to(device=device, dtype=dtype)  # type: ignore[no-any-return, attr-defined]

    def unwrapModel(self, *, system: object) -> nn.Module:
        module = getattr(system, "module", None)
        if isinstance(module, nn.Module):
            return module
        if isinstance(system, nn.Module):
            return system
        raise TypeError(f"Expected system to expose .module nn.Module, got {type(system).__name__}")

    def checkpointPayload(self, *, target: ExperimentTargetConfig, train: TrainConfig, tokenizer_file: str) -> dict[str, Any]:
        return {
            "target": str(target.name),
            "trainer": {"kind": "diffusion_codegen", "timesteps": int(self.settings.timesteps), "schedule": str(self.settings.schedule)},
            "train": json.loads(train.model_dump_json()),
            "tokenizer_file": str(tokenizer_file),
            "system_config": dict(target.system.config),
            "data_config": dict(target.data.config),
        }

