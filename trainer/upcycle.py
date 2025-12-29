"""Upcycling trainer (target-based).

Upcycling is a teacher/student workflow. In the manifest schema, an experiment
target provides:
- `system.language_model` config (student architecture)
- `runs[]` with `train.phase` in {blockwise, global}
- `train.teacher_ckpt` pointing at the pretrained teacher weights

Internally we reuse the existing blockwise/global steppers, collector, verifier,
and checkpointer components. The only "bridge" is a small manifest shim that
exposes `model` for the initializer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import re
import torch

from carmath import autocast_dtype_str, weight_dtype, weight_dtype_str, token_budget_batch_size
from config.defaults import Defaults
from config.group import Group
from config.manifest import Manifest
from config.model import ModelConfig
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from config.collector import DefaultCollectorConfig
from config.checkpointer import DefaultCheckPointerConfig
from config.initializer import DefaultInitializerConfig
from config.verifier import DefaultVerifierConfig
from config.stepper import DefaultStepperConfig
from console import logger
from instrumentation import Instrumentation, generate_analysis_png
from config.instrumentation import (
    InstrumentationConfig,
    TensorBoardConfig,
    WandBConfig,
    LivePlotConfig,
    JSONLConfig,
)
from runtime import RuntimePlan, load_plan, make_plan_key, save_plan
from trainer.upcycle_init_context import UpcycleInitContext
from trainer.upcycle_context import UpcycleContext
from trainer.initializers import Initializer
from trainer.collectors import Collector
from trainer.verifiers import Verifier
from trainer.checkpointers import CheckPointer
from trainer.steppers import Stepper
from data import build_token_dataset
from runtime.tensordict_utils import TensorDictBase, collate_tensordict
from torch.utils.data import DataLoader
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class _ManifestShim:
    """Minimal shape expected by legacy upcycling components."""

    name: str | None
    notes: str
    defaults: Defaults
    model: ModelConfig


class UpcycleTrainer:
    """Target-based entrypoint for the upcycling pipeline."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | None = None,
        resume_from: str | None = None,
    ) -> None:
        self._checkpoint_dir_override = checkpoint_dir
        self._resume_from = resume_from

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: Any,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            return None

        if target.system.ref != "system.language_model":
            raise ValueError("UpcycleTrainer currently requires system.ref=system.language_model")
        model_payload = target.system.config.get("model", None)
        if not isinstance(model_payload, dict):
            raise ValueError("system.language_model requires system.config.model (dict)")

        model_cfg = ModelConfig.model_validate(model_payload)
        shim = _ManifestShim(
            name=manifest.name,
            notes=str(manifest.notes or ""),
            defaults=manifest.defaults,
            model=model_cfg,
        )

        # Build a legacy Group object for the existing collector/checkpointer layout.
        data_path = target.data.config.get("path", "")
        if not isinstance(data_path, str):
            data_path = str(data_path)
        group = Group(
            name=str(target.name),
            description=str(getattr(target, "description", "")),
            data=str(data_path),
            runs=list(target.runs),
            benchmarks=target.benchmarks,
        )

        # Initialize session once per target (teacher/student live across runs).
        init_train = self._find_init_train(group)
        session = _UpcycleSession(
            shim_manifest=shim,
            group=group,
            train=init_train,
            checkpoint_dir=self._checkpoint_dir_override,
            resume_from=self._resume_from,
        )
        session.run_all()
        return {
            "teacher": session.teacher,
            "student": session.student,
            "device": session.device,
            "checkpoint_dir": session.checkpoint_dir,
        }

    @staticmethod
    def _find_init_train(group: Group) -> TrainConfig:
        for r in group.runs:
            if r.train is not None:
                return r.train
        raise ValueError(f"Target '{group.name}' has no runs with train config.")


class _UpcycleSession:
    """A single upcycling session (teacher/student shared across runs)."""

    def __init__(
        self,
        *,
        shim_manifest: _ManifestShim,
        group: Group,
        train: TrainConfig,
        checkpoint_dir: str | None,
        resume_from: str | None,
    ) -> None:
        self.manifest = shim_manifest
        self.group = group
        self.defaults = shim_manifest.defaults
        self.model_cfg = shim_manifest.model

        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else Path("runs") / str(group.name)
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every = int(getattr(self.defaults.runtime, "save_every", 100))

        self.device = torch.device(train.device)
        self.runtime_plan = self._load_or_create_runtime_plan(train)
        self.dtype = weight_dtype(self.device, str(self.runtime_plan.dtype))

        self.inst: Instrumentation | None = None
        self._resume_state: dict[str, object] | None = None

        self.initializer: Initializer = cast(Initializer, DefaultInitializerConfig().build())
        self.teacher, self.student = self.initializer.init_models(
            train,
            UpcycleInitContext(
                manifest=cast(object, self.manifest),
                group=self.group,
                defaults=self.defaults,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                dtype=self.dtype,
                runtime_plan=self.runtime_plan,
                dist_ctx=None,
            ),
        )

        self._teacher_sanity_check(train)

        self.verifier: Verifier = cast(Verifier, DefaultVerifierConfig().build())
        self.collector: Collector = cast(Collector, DefaultCollectorConfig().build())
        self.checkpointer: CheckPointer = cast(CheckPointer, DefaultCheckPointerConfig().build())
        self.stepper: Stepper = cast(Stepper, DefaultStepperConfig().build())

        if resume_from is not None:
            try:
                self._resume_state = self.checkpointer.load_resume(ctx=self._ctx(), path=Path(resume_from))
            except Exception:
                logger.warning("Failed to load resume state, continuing without resume")
                self._resume_state = None

    def _teacher_sanity_check(self, train: TrainConfig) -> None:
        """Fail fast if the teacher or dataset is obviously broken."""
        if train.teacher_ckpt is None:
            logger.error("Teacher sanity check skipped: teacher_ckpt is required")
            return
        if not bool(getattr(train, "teacher_sanity_check", True)):
            logger.error("Teacher sanity check skipped: teacher_sanity_check is disabled")
            return

        # Use the target's configured dataset path (groups[].data).
        data_path = str(self.group.data)
        try:
            dataset = build_token_dataset(path=data_path, block_size=int(train.block_size))
        except Exception as e:
            logger.error(f"Teacher sanity check skipped: failed to load dataset {data_path!r}: {e}")
            return

        loader: DataLoader[TensorDictBase] = DataLoader(
            dataset,
            batch_size=int(getattr(train, "teacher_sanity_batch_size", 1)),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_tensordict,
        )

        # Resolve vocab size from teacher embeddings (tied or untied).
        try:
            from benchmark.utils import get_model_vocab_size

            vocab_size = int(get_model_vocab_size(self.teacher, default=32000))
        except Exception:
            logger.error("Teacher sanity check skipped: failed to get model vocab size, setting to default 32000")
            vocab_size = 32000

        max_batches = int(getattr(train, "teacher_sanity_batches", 2))
        max_nll = float(getattr(train, "teacher_sanity_max_nll", 20.0))

        self.teacher.eval()
        total_loss = 0.0
        total_tokens = 0
        n = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["input_ids"].to(self.device)
                y = batch["target_ids"].to(self.device)

                # Token/vocab compatibility check.
                mx = int(torch.maximum(x.max(), y.max()).item())
                if mx >= int(vocab_size):
                    raise ValueError(
                        f"Teacher sanity check failed: dataset token IDs exceed teacher vocab "
                        f"(max_id={mx}, vocab_size={vocab_size}). "
                        "This usually means the dataset was tokenized with a different tokenizer than the teacher."
                    )

                logits = self.teacher(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )
                total_loss += float(loss)
                total_tokens += int(y.numel())
                n += 1
                if n >= max_batches:
                    break

        denom = max(1, int(total_tokens))
        nll = float(total_loss) / float(denom)
        logger.key_value({"teacher_sanity_nll": f"{nll:.6f}", "teacher_sanity_tokens": str(total_tokens)})

        if nll > max_nll:
            raise ValueError(
                f"Teacher sanity check failed: teacher NLL={nll:.6f} exceeds max_nll={max_nll:.6f}. "
                "This commonly indicates a bad checkpoint load or tokenizer/dataset mismatch."
            )

    def run_all(self) -> None:
        logger.header("Experiment", str(self.group.name))
        logger.info(str(self.group.description))

        for i, run in enumerate(self.group.runs):
            phase = run.train.phase.value if run.train else "unknown"
            logger.step(i + 1, len(self.group.runs), f"Run '{run.id}' ({phase})")
            self.run_one(run)

    def run_one(self, run: Run) -> None:
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        if run.train.phase not in (TrainPhase.BLOCKWISE, TrainPhase.GLOBAL):
            raise ValueError(f"UpcycleTrainer only supports blockwise/global phases, got {run.train.phase}")

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
            logger.error("Failed to generate analysis PNG, continuing")

    def _ctx(self) -> UpcycleContext:
        return UpcycleContext(
            manifest=cast(object, self.manifest),
            group=self.group,
            defaults=self.defaults,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
            dtype=self.dtype,
            runtime_plan=self.runtime_plan,
            teacher=self.teacher,
            student=self.student,
            inst=self.inst,
            dist_ctx=None,
        )

    def _build_instrumentation(self, run: Run) -> Instrumentation:
        instrument = str(getattr(self.defaults.logging, "instrument", "rich"))
        tokens = {t for t in re.split(r"[,+\s]+", instrument.lower()) if t}

        configs: list[InstrumentationConfig] = [
            JSONLConfig(out_dir=str(self.checkpoint_dir), filename="train.jsonl"),
        ]
        if "tb" in tokens or "tensorboard" in tokens:
            configs.append(TensorBoardConfig(out_dir=str(self.checkpoint_dir / "tb" / str(run.id))))
        if "live" in tokens or "plot" in tokens or "liveplot" in tokens:
            configs.append(LivePlotConfig(title=f"{self.group.name}:{run.id}"))

        if bool(getattr(self.defaults.logging, "wandb", False)):
            configs.append(
                WandBConfig(
                    project=str(getattr(self.defaults.logging, "wandb_project", "")),
                    entity=str(getattr(self.defaults.logging, "wandb_entity", "") or ""),
                    run_name=f"{self.group.name}:{run.id}",
                    group=str(self.group.name),
                )
            )

        return Instrumentation(configs, self.checkpoint_dir)

    def _load_or_create_runtime_plan(self, train: TrainConfig) -> RuntimePlan:
        train_payload = train.model_dump()
        train_payload.pop("teacher_ckpt", None)
        payload: dict[str, Any] = {
            "device": str(self.device),
            "torch": str(getattr(torch, "__version__", "")),
            "model": self.model_cfg.model_dump(),
            "train": train_payload,
        }
        key = make_plan_key(payload)
        plan_path = self.checkpoint_dir / "plans" / f"{key}.json"
        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            return existing

        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            dtype_str = weight_dtype_str(self.device)

        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            amp_dtype_str = autocast_dtype_str(self.device)

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

        plan = RuntimePlan(
            key=key,
            device=str(self.device),
            torch_version=str(getattr(torch, "__version__", "")),
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=int(batch_size),
            compile=bool(getattr(train, "compile_model", False)),
            compile_mode=str(getattr(train, "compile_mode", "reduce-overhead")),
        )
        try:
            save_plan(plan_path, plan, payload=payload)
        except Exception:
            logger.error("Failed to save runtime plan, continuing")
        return plan

