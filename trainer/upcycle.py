"""Model upcycling: converting pretrained models to new architectures.

Upcycling takes a pretrained model (like Llama) and trains it to use a new
architecture (like DBA attention) while preserving its learned knowledge.
We do this by distillation: the original model is the "teacher" and the
new architecture is the "student." The student learns to produce the same
outputs as the teacher, then we fine-tune on language modeling to recover
any lost performance.
"""
from __future__ import annotations

from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, cast

import re
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from carmath import token_budget_batch_size
from carmath import autocast_dtype_str, weight_dtype, weight_dtype_str
from config.defaults import Defaults
from config.group import Group
from config.manifest import Manifest
from config.run import Run
from config.train import TrainConfig, TrainPhase
from config.collector import DefaultCollectorConfig
from config.checkpointer import DefaultCheckPointerConfig
from config.initializer import DefaultInitializerConfig
from config.verifier import DefaultVerifierConfig
from console import logger
from instrumentation import Instrumentation, generate_analysis_png
from config.instrumentation import (
    TensorBoardConfig,
    WandBConfig,
    LivePlotConfig,
    JSONLConfig,
)
from layer.attention import AttentionLayer
from trainer.upcycle_init_context import UpcycleInitContext
from trainer.initializers import Initializer
from runtime import RuntimePlan, load_plan, make_plan_key, save_plan
from config.stepper import DefaultStepperConfig
from trainer.upcycle_context import UpcycleContext
from trainer.collectors import Collector
from trainer.verifiers import Verifier
from trainer.checkpointers import CheckPointer
from trainer.steppers import Stepper
from trainer.distill import DistillLoss
from trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler


class Upcycle:
    """Orchestrates the full upcycling pipeline: distillation, fine-tuning, verification.

    The pipeline has two main training phases:
    1. Blockwise distillation: Train each attention layer individually to match the teacher
    2. Global fine-tuning: Train the whole model on language modeling loss

    After training, verification checks that the student produces outputs similar
    to the teacher, catching training failures before expensive benchmarking.
    """

    def __init__(
        self,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
        *,
        dist_config: DistributedConfig | None = None,
        defaults: Defaults | None = None,
        checkpoint_dir: Path | str | None = None,
        resume_from: Path | str | None = None,
    ) -> None:
        """Initialize the upcycling trainer.

        This loads both the teacher (original architecture) and student (new
        architecture), applies the pretrained weights, and sets up distributed
        training if configured.

        Args:
            manifest: Model architecture specification
            group: Experiment group with data paths and settings
            train: Training hyperparameters
            dist_config: Optional distributed training settings
            defaults: Optional global defaults (save frequency, etc.)
            checkpoint_dir: Where to save checkpoints
            resume_from: Path to resume training from a checkpoint
        """
        self.manifest = manifest
        self.group = group
        self.defaults = defaults

        self.save_every = defaults.save_every if defaults else 500
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("runs") / group.name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.inst: Instrumentation | None = None
        self._resume_state: dict[str, object] | None = None

        # Set up distributed training if configured
        self.dist_config = dist_config
        self.dist_ctx: DistributedContext | None = None
        if dist_config is not None and dist_config.strategy != DistributedStrategy.NONE:
            self.dist_ctx = DistributedContext.init(dist_config)
            self.device = self.dist_ctx.device
        else:
            self.device = self.parse_device(train.device)

        self.device_name = str(self.device)
        self.runtime_plan = self._load_or_create_runtime_plan(train)
        self.dtype = self.parse_dtype(self.runtime_plan.dtype)
        self.initializer: Initializer = cast(Initializer, DefaultInitializerConfig().build())
        self.teacher, self.student = self.initializer.init_models(
            train,
            UpcycleInitContext(
                manifest=self.manifest,
                group=self.group,
                defaults=self.defaults,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                dtype=self.dtype,
                runtime_plan=self.runtime_plan,
                dist_ctx=self.dist_ctx,
            ),
        )
        self.verifier: Verifier = cast(Verifier, DefaultVerifierConfig().build())
        self.collector: Collector = cast(Collector, DefaultCollectorConfig().build())
        self.checkpointer: CheckPointer = cast(CheckPointer, DefaultCheckPointerConfig().build())
        self.stepper: Stepper = cast(Stepper, DefaultStepperConfig().build())

        if resume_from is not None:
            self._resume_state = self.checkpointer.load_resume(ctx=self._ctx(), path=Path(resume_from))

    def run(self, run: Run) -> None:
        """Execute a single training run (blockwise or global phase).

        After training completes, runs any configured verification steps.
        """
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")

        torch.manual_seed(run.seed)
        self.inst = self._build_instrumentation(run)

        self.stepper.run(
            run,
            self._ctx(),
            collector=self.collector,
            checkpointer=self.checkpointer,
            save_every=int(self.save_every),
            resume_state=self._resume_state,
        )

        self.verifier.verify(run, self._ctx())
        if self.inst is not None:
            self.inst.close()
            self.inst = None
        try:
            generate_analysis_png(
                self.checkpoint_dir / "train.jsonl",
                self.checkpoint_dir / f"{run.id}_analysis.png",
            )
        except Exception:
            pass

    def _ctx(self) -> UpcycleContext:
        return UpcycleContext(
            manifest=self.manifest,
            group=self.group,
            defaults=self.defaults,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
            dtype=self.dtype,
            runtime_plan=self.runtime_plan,
            teacher=self.teacher,
            student=self.student,
            inst=self.inst,
            dist_ctx=self.dist_ctx,
        )

    def _build_instrumentation(self, run: Run) -> Instrumentation:
        """Build instrumentation backends from defaults."""
        from config.instrumentation import InstrumentationConfig

        instrument = str(getattr(self.defaults, "instrument", "rich") if self.defaults else "rich")
        tokens = {t for t in re.split(r"[,+\s]+", instrument.lower()) if t}

        configs: list[InstrumentationConfig] = [
            JSONLConfig(out_dir=str(self.checkpoint_dir), filename="train.jsonl"),
        ]
        if "tb" in tokens or "tensorboard" in tokens:
            configs.append(TensorBoardConfig(out_dir=str(self.checkpoint_dir / "tb" / str(run.id))))
        if "live" in tokens or "plot" in tokens or "liveplot" in tokens:
            configs.append(LivePlotConfig(title=f"{self.group.name}:{run.id}"))
        if self.defaults and getattr(self.defaults, "wandb", False):
            configs.append(WandBConfig(
                project=str(getattr(self.defaults, "wandb_project", "")),
                entity=str(getattr(self.defaults, "wandb_entity", "") or ""),
                run_name=f"{self.group.name}:{run.id}",
                group=self.group.name,
            ))
        return Instrumentation(configs, self.checkpoint_dir)

    def _log(self, msg: str) -> None:
        """Log only from the main process in distributed mode."""
        if self.dist_ctx is not None:
            self.dist_ctx.log(msg)
        else:
            logger.info(msg)

    def require_train(self, run: Run) -> TrainConfig:
        """Get the train config or raise if missing."""
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        return run.train

    @staticmethod
    def parse_device(device: str) -> torch.device:
        """Parse a device string like 'cuda', 'mps', or 'cpu'."""
        return torch.device(device)

    def parse_dtype(self, dtype: str) -> torch.dtype:
        """Parse a dtype string to a torch dtype.

        Supports an "auto" mode that picks a sane default based on the device.
        """
        return weight_dtype(self.device, str(dtype))

    def _load_or_create_runtime_plan(self, train: TrainConfig) -> RuntimePlan:
        """Derive a runtime plan and persist it for reuse."""

        # Build a stable payload that excludes volatile fields like checkpoint paths.
        train_payload = train.model_dump()
        train_payload.pop("teacher_ckpt", None)
        payload: dict[str, Any] = {
            "device": str(self.device),
            "torch": str(getattr(torch, "__version__", "")),
            "model": self.manifest.model.model_dump(),
            "train": train_payload,
        }
        key = make_plan_key(payload)
        plan_path = self.checkpoint_dir / "plans" / f"{key}.json"
        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            return existing

        # Resolve decisions.
        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            dtype_str = weight_dtype_str(self.device)

        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            amp_dtype_str = autocast_dtype_str(self.device)

        # Batch size tuning decision.
        batch_size = int(train.batch_size)
        if bool(getattr(train, "auto_batch_size", False)):
            ref = int(getattr(train, "auto_batch_ref_block_size", 512))
            min_bs = int(getattr(train, "auto_batch_min", 1))
            batch_size = token_budget_batch_size(
                batch_size,
                block_size=int(train.block_size),
                ref_block_size=int(ref),
                min_batch_size=int(min_bs),
            )

        # Compile decision.
        compile_setting: object = getattr(train, "compile_model", False)
        compile_mode = str(getattr(train, "compile_mode", "reduce-overhead"))
        should_compile = False
        if isinstance(compile_setting, bool):
            should_compile = compile_setting
        else:
            s = str(compile_setting).strip().lower()
            if s == "auto":
                should_compile = self.device.type == "cuda"
            elif s in ("1", "true", "yes", "on"):
                should_compile = True
            else:
                should_compile = False

        plan = RuntimePlan(
            key=key,
            device=str(self.device),
            torch_version=str(getattr(torch, "__version__", "")),
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=int(batch_size),
            compile=bool(should_compile),
            compile_mode=str(compile_mode),
        )
        try:
            save_plan(plan_path, plan, payload=payload)
        except Exception:
            pass
        return plan

    @staticmethod
    def _int_or(value: object, default: int = 0) -> int:
        """Best-effort int conversion for checkpoint metadata."""
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(default)
