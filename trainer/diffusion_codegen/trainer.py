"""Diffusion codegen trainer (manifest-driven)

This trainer orchestrates a diffusion-on-embeddings recipe for code generation.
The implementation is intentionally split into small modules to keep each file
and method within Caramba's size limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.console import logger
from caramba.trainer.diffusion_codegen.generation import GenerationRunner, GenerationSettings
from caramba.trainer.diffusion_codegen.optim import OptimizerFactory
from caramba.trainer.diffusion_codegen.tokenizer_ops import TokenizerManager
from caramba.trainer.diffusion_codegen.train_loader import LoaderFactory
from caramba.trainer.diffusion_codegen.training import TrainingRunner, TrainingSettings


@dataclass
class DiffusionCodegenTrainer:
    """Diffusion codegen trainer

    Config:
    - action: "train" or "generate"
    - checkpoint_dir: optional override for `runs/<target.name>/`
    - tokenizer: tokenizer policy/training settings
    - sampler: sampler settings shared by train/generate
    - generate: generation settings (checkpoint selection, prompt, sample count)
    """

    action: str = "train"
    checkpoint_dir: str | None = None

    timesteps: int = 1000
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    mse_lambda: float = 0.5
    unconditional_prob: float = 0.1
    self_condition_prob: float = 0.5
    grad_clip: float = 1.0

    use_ema: bool = False
    ema_decay: float = 0.999

    sampler: dict[str, Any] | None = None
    tokenizer: dict[str, Any] | None = None
    generate: dict[str, Any] | None = None

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: Any,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            logger.info("Dry run requested, skipping diffusion_codegen")
            return None

        action = self.normalizedAction()
        checkpoint_dir = self.resolveCheckpointDir(target_name=str(target.name))
        tokenizer_file = TokenizerManager(config=self.tokenizer).ensureTokenizerFile()
        self.applyTokenizerPaths(target=target, tokenizer_file=tokenizer_file)

        if action == "train":
            return self.runTrain(manifest=manifest, target=target, engine=engine, checkpoint_dir=checkpoint_dir, tokenizer_file=tokenizer_file)
        return self.runGenerate(target=target, engine=engine, checkpoint_dir=checkpoint_dir, tokenizer_file=tokenizer_file)

    def normalizedAction(self) -> str:
        action = str(self.action).lower().strip()
        if action not in {"train", "generate"}:
            raise ValueError(f"Unknown trainer action: {self.action!r}")
        return action

    def runTrain(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: Any,
        checkpoint_dir: Path,
        tokenizer_file: str,
    ) -> dict[str, Any]:
        dataset_comp = engine.registry.build(target.data, backend=str(target.backend))
        system = engine.registry.build(target.system, backend=str(target.backend))
        TrainingRunner(
            settings=self.trainingSettings(),
            optimizerFactory=OptimizerFactory(),
            loaderFactory=LoaderFactory(),
        ).runTarget(
            manifest=manifest,
            target=target,
            dataset_comp=dataset_comp,
            system=system,
            checkpoint_dir=checkpoint_dir,
            tokenizer_file=tokenizer_file,
        )
        return {"checkpoint_dir": checkpoint_dir}

    def runGenerate(
        self,
        *,
        target: ExperimentTargetConfig,
        engine: Any,
        checkpoint_dir: Path,
        tokenizer_file: str,
    ) -> dict[str, Any]:
        artifacts = GenerationRunner(settings=self.generationSettings()).run(
            target=target,
            engine=engine,
            checkpoint_dir=checkpoint_dir,
            tokenizer_file=tokenizer_file,
        )
        return {"artifacts": artifacts, "checkpoint_dir": checkpoint_dir}

    def resolveCheckpointDir(self, *, target_name: str) -> Path:
        root = Path(self.checkpoint_dir) if self.checkpoint_dir else Path("runs") / str(target_name)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def trainingSettings(self) -> TrainingSettings:
        return TrainingSettings(
            timesteps=int(self.timesteps),
            schedule=str(self.schedule),
            beta_start=float(self.beta_start),
            beta_end=float(self.beta_end),
            mse_lambda=float(self.mse_lambda),
            unconditional_prob=float(self.unconditional_prob),
            self_condition_prob=float(self.self_condition_prob),
            grad_clip=float(self.grad_clip),
            use_ema=bool(self.use_ema),
            ema_decay=float(self.ema_decay),
            sampler=self.sampler,
        )

    def generationSettings(self) -> GenerationSettings:
        return GenerationSettings(
            timesteps=int(self.timesteps),
            schedule=str(self.schedule),
            beta_start=float(self.beta_start),
            beta_end=float(self.beta_end),
            sampler=self.sampler,
            generate=self.generate,
        )

    def applyTokenizerPaths(self, *, target: ExperimentTargetConfig, tokenizer_file: str) -> None:
        target.data.config.setdefault("tokenizer_file", str(tokenizer_file))
        target.system.config.setdefault("tokenizer_file", str(tokenizer_file))

