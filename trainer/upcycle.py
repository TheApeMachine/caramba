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
import os
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
        # Preserve the full data.config payload so we can run manifest-driven dataset prep.
        # Normalize to dict[str, object] for internal use (pyright-friendly).
        if isinstance(target.data.config, dict):
            data_config: dict[str, object] = {str(k): v for k, v in target.data.config.items()}
        else:
            data_config = {"path": str(data_path)}
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
            data_config=data_config,
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
        data_config: dict[str, object],
        train: TrainConfig,
        checkpoint_dir: str | None,
        resume_from: str | None,
    ) -> None:
        self.manifest = shim_manifest
        self.group = group
        self.data_config = dict(data_config)
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
        prepare_cfg = self.data_config.get("prepare", None)
        if prepare_cfg is not None and not isinstance(prepare_cfg, dict):
            prepare_cfg = None

        def _maybe_prepare(*, overwrite: bool) -> None:
            if prepare_cfg is None:
                return
            kind = str(prepare_cfg.get("type", "fineweb")).lower().strip()
            if kind not in {"fineweb", "fineweb_npy"}:
                raise ValueError(f"Unsupported data.config.prepare.type={kind!r}")
            # Import locally so training can run without dataset deps unless requested.
            from prepare_fineweb import prepare_fineweb_npy

            # Manifest-driven defaults: fall back to manifest defaults.data.tokenizer.
            tok = str(prepare_cfg.get("tokenizer", getattr(self.defaults.data, "tokenizer", "llama")))
            # Default model_id: use teacher checkpoint for llama unless overridden.
            model_id = str(prepare_cfg.get("model_id", train.teacher_ckpt or "meta-llama/Llama-3.2-1B"))
            if model_id.startswith("hf://"):
                model_id = model_id[5:]
            ds_id = str(prepare_cfg.get("dataset", "HuggingFaceFW/fineweb"))
            subset = prepare_cfg.get("subset", None)
            subset_s = str(subset) if isinstance(subset, str) and subset else None
            split = str(prepare_cfg.get("split", "train"))
            text_field = str(prepare_cfg.get("text_field", "text"))
            max_chars = int(prepare_cfg.get("max_chars", 50_000))
            append_eos = bool(prepare_cfg.get("append_eos", True))
            append_bos = bool(prepare_cfg.get("append_bos", False))
            token_budget = str(prepare_cfg.get("tokens", "100M"))
            logger.info(
                f"Preparing dataset (manifest-driven): {ds_id} split={split} tokenizer={tok} → {data_path}"
            )
            prepare_fineweb_npy(
                tokens=token_budget,
                output=data_path,
                tokenizer=tok,
                model_id=model_id,
                dataset=ds_id,
                subset=subset_s,
                split=split,
                text_field=text_field,
                max_chars=max_chars,
                append_eos=append_eos,
                append_bos=append_bos,
                overwrite=bool(overwrite),
            )
        try:
            dataset = build_token_dataset(path=data_path, block_size=int(train.block_size))
        except Exception as e:
            # If the dataset doesn't exist yet, and a prepare config is present, build it now.
            if prepare_cfg is not None:
                _maybe_prepare(overwrite=False)
                dataset = build_token_dataset(path=data_path, block_size=int(train.block_size))
            else:
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
        max_ppl = float(getattr(train, "teacher_sanity_max_ppl", 200.0))
        ref_kind = str(getattr(train, "teacher_sanity_reference", "none")).lower().strip()
        ref_batches = int(getattr(train, "teacher_sanity_ref_batches", 1))
        max_ppl_ratio = float(getattr(train, "teacher_sanity_max_ppl_ratio_vs_ref", 1.25))
        max_nll_delta = float(getattr(train, "teacher_sanity_max_nll_delta_vs_ref", 0.25))
        ref_fail_fast = bool(getattr(train, "teacher_sanity_reference_fail_fast", True))

        self.teacher.eval()
        total_loss = 0.0
        total_tokens = 0
        n = 0
        # Teacher parity should be a *correctness* check, not a kernel benchmark.
        # Metal fast-path kernels are great for speed, but if they have a bug we want
        # to detect it. For the sanity check, force pure PyTorch paths even on MPS.
        old_disable_all = os.environ.get("CARAMBA_DISABLE_METAL_KERNELS")
        os.environ["CARAMBA_DISABLE_METAL_KERNELS"] = "1"
        try:
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
                    if not torch.isfinite(logits).all():
                        raise ValueError(
                            "Teacher sanity check failed: teacher produced NaN/Inf logits. "
                            "This usually indicates a bad checkpoint load, dtype/device issue, or numerical instability."
                        )
                    loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction="sum",
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Teacher sanity check failed: teacher loss is NaN/Inf.")
                    total_loss += float(loss)
                    total_tokens += int(y.numel())
                    n += 1
                    if n >= max_batches:
                        break
        finally:
            if old_disable_all is None:
                os.environ.pop("CARAMBA_DISABLE_METAL_KERNELS", None)
            else:
                os.environ["CARAMBA_DISABLE_METAL_KERNELS"] = old_disable_all

        denom = max(1, int(total_tokens))
        nll = float(total_loss) / float(denom)
        try:
            ppl = float(torch.exp(torch.tensor(nll)).item())
        except (OverflowError, RuntimeError) as e:
            logger.warning(f"Failed to compute perplexity from nll={nll:.6f}: {type(e).__name__}: {e}")
            ppl = float("inf")
        logger.key_value(
            {
                "teacher_sanity_nll": f"{nll:.6f}",
                "teacher_sanity_ppl": f"{ppl:.3f}",
                "teacher_sanity_tokens": str(total_tokens),
            }
        )

        if nll > max_nll:
            raise ValueError(
                f"Teacher sanity check failed: teacher NLL={nll:.6f} exceeds max_nll={max_nll:.6f}. "
                "This commonly indicates a bad checkpoint load or tokenizer/dataset mismatch."
            )

        # Optional gold-reference check: compare against transformers on the same tokens.
        if ref_kind == "hf":
            if train.teacher_ckpt is None or not str(train.teacher_ckpt).startswith("hf://"):
                logger.warning("teacher_sanity_reference=hf requested, but teacher_ckpt is not hf://...; skipping reference check.")
            else:
                try:
                    from transformers import AutoModelForCausalLM  # type: ignore
                except Exception as e:
                    if ref_fail_fast:
                        raise
                    logger.warning(f"teacher_sanity_reference=hf requested, but transformers import failed: {e}")
                else:
                    repo_id = str(train.teacher_ckpt)[5:]
                    hf_device = torch.device("cpu")
                    try:
                        hf_any = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=False)  # type: ignore[call-arg]
                        hf = cast(torch.nn.Module, hf_any)
                        hf.eval()
                        hf.to(hf_device)
                        hf_total = 0.0
                        hf_tokens = 0
                        hf_n = 0
                        with torch.no_grad():
                            for batch in loader:
                                x = batch["input_ids"].to(device=hf_device)
                                y = batch["target_ids"].to(device=hf_device)
                                out = hf(input_ids=x)  # type: ignore[call-arg]
                                logits_hf = getattr(out, "logits", None)
                                if logits_hf is None:
                                    logits_hf = out[0] if isinstance(out, (tuple, list)) else out
                                loss_hf = F.cross_entropy(
                                    logits_hf.float().view(-1, logits_hf.size(-1)),
                                    y.view(-1),
                                    reduction="sum",
                                )
                                hf_total += float(loss_hf)
                                hf_tokens += int(y.numel())
                                hf_n += 1
                                if hf_n >= max(1, ref_batches):
                                    break
                        hf_nll = float(hf_total) / float(max(1, hf_tokens))
                        hf_ppl = float(torch.exp(torch.tensor(hf_nll)).item())
                        ppl_ratio = float(ppl) / float(hf_ppl) if hf_ppl > 0 else float("inf")
                        nll_delta = float(nll) - float(hf_nll)
                        logger.key_value(
                            {
                                "teacher_ref_hf_nll": f"{hf_nll:.6f}",
                                "teacher_ref_hf_ppl": f"{hf_ppl:.3f}",
                                "teacher_vs_hf_ppl_ratio": f"{ppl_ratio:.3f}",
                                "teacher_vs_hf_nll_delta": f"{nll_delta:.6f}",
                            }
                        )
                        if (ppl_ratio > max_ppl_ratio) or (abs(nll_delta) > max_nll_delta):
                            raise ValueError(
                                "Teacher sanity check failed: caramba teacher does not match HF reference on the same tokens. "
                                f"ppl_ratio={ppl_ratio:.3f} (max {max_ppl_ratio:.3f}), "
                                f"nll_delta={nll_delta:.6f} (max ±{max_nll_delta:.6f}). "
                                "This indicates a teacher weight-loading or forward-parity bug (not a dataset issue)."
                            )
                    except Exception as e:
                        if ref_fail_fast:
                            raise
                        logger.warning(f"teacher_sanity_reference=hf failed (skipping): {e}")

        if ppl > max_ppl:
            # Optional self-healing: if a manifest prepare config is present, regenerate the dataset
            # once (overwrite) and retry the sanity check. This keeps the workflow fully
            # manifest-driven: users run one command and the manifest declares how to produce data.
            if prepare_cfg is not None and bool(prepare_cfg.get("rebuild_on_failure", True)):
                logger.warning(
                    f"Teacher sanity failed (ppl={ppl:.3f} > {max_ppl:.3f}); rebuilding dataset per manifest and retrying once..."
                )
                _maybe_prepare(overwrite=True)
                # Re-run once with rebuild disabled to avoid loops.
                prepare_cfg["rebuild_on_failure"] = False
                return self._teacher_sanity_check(train)

            raise ValueError(
                f"Teacher sanity check failed: teacher perplexity={ppl:.3f} exceeds max_ppl={max_ppl:.3f}. "
                "This commonly indicates the dataset tokens do not match the teacher tokenizer "
                "(e.g. tiktoken tokens fed to a Llama tokenizer), even if token IDs are in-range."
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

        # Manifest-driven auto-resume / skip-if-final.
        auto_resume = bool(getattr(run.train, "auto_resume", True))
        skip_if_final = bool(getattr(run.train, "skip_if_final", True))
        phase_name = "blockwise" if run.train.phase == TrainPhase.BLOCKWISE else "global"

        if auto_resume:
            try:
                latest = self.checkpointer.latest(ctx=self._ctx(), run_id=str(run.id), phase=phase_name)
            except Exception:
                latest = None
            if latest is not None and latest.exists():
                # If a final checkpoint exists, optionally skip this phase entirely.
                if skip_if_final and latest.name.endswith("_final.pt"):
                    logger.warning(f"Skipping run '{run.id}' ({phase_name}): found final checkpoint {latest}")
                    try:
                        self._resume_state = self.checkpointer.load_resume(ctx=self._ctx(), path=latest)
                    except Exception:
                        # Best-effort: even if resume parsing fails, continue to next run.
                        logger.warning("Failed to load final checkpoint state, continuing")
                    return
                # Otherwise resume from the latest checkpoint and continue.
                if self._resume_state is None:
                    try:
                        self._resume_state = self.checkpointer.load_resume(ctx=self._ctx(), path=latest)
                    except Exception:
                        logger.warning("Failed to auto-resume checkpoint, continuing without resume")
                        self._resume_state = None

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

