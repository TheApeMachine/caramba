"""Standard trainer (target-based).

This trainer is objective-driven:
- dataset provides batches
- system produces outputs
- objective computes loss from (batch, outputs)

No assumptions about "tokens" are baked into the trainer beyond the chosen
components.
"""
from __future__ import annotations

import inspect
import json
import math
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from torch.profiler import ProfilerActivity, profile

from carmath import (
    autocast_dtype,
    global_grad_norm_l2,
    safe_perplexity_from_nll,
)
from compiler.plan import Planner
from config.defaults import Defaults
from config.manifest import Manifest
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from console import logger
from instrumentation import RunLogger
try:
    # Newer builds export UAA context alongside MOSAIC viz context.
    from instrumentation.viz import TrainingUAAContext as _TrainingUAAContext, TrainingVizMosaicContext
    _HAS_UAA_CTX = True
except Exception:  # pragma: no cover
    # Backward-compat: older deployments may not have UAA context in instrumentation.viz.
    from instrumentation.viz import TrainingVizMosaicContext  # type: ignore
    _TrainingUAAContext = None  # type: ignore[assignment]
    _HAS_UAA_CTX = False
from instrumentation.training_metrics import update_training_metrics
from instrumentation.wandb_writer import WandBWriter
from layer.attention import AttentionLayer
from layer.memory_block.block import MemoryBlockLayer
from layer.memory_block.memory.tuner import reset_shared_tuner
from runtime.plan import RuntimePlan
from runtime.tensordict_utils import (
    TensorDictBase,
    as_tensordict,
    to_device,
)
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from trainer.mosaic_table2 import Table2SummaryWriter, Table2Telemetry
from trainer.swap_manager import SwapManager
from trainer.objectives import MosaicNextTokenWithAuxObjective
from trainer.uaa import UAAState
from trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)
from optimizer.adamw_master import AdamWMaster
from optimizer.lion import Lion
from runtime.plan.builder import RuntimePlanBuilder
from trainer.train_dataloader.builder import TrainDataLoaderBuilder


class _Engine(Protocol):
    registry: Any


def _format_state_keys_sample(keys: list[str], *, limit: int = 25) -> str:
    if not keys:
        return ""
    n = len(keys)
    k = [str(x) for x in keys[: max(0, min(limit, n))]]
    out = "\n".join(f"  - {x}" for x in k)
    if n > len(k):
        out += f"\n  ... (+{n - len(k)} more)"
    return out


def _bytes_to_mb(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)


def _is_unexpected_ctx_kwarg_error(e: TypeError) -> bool:
    msg = str(e)
    return ("unexpected keyword argument" in msg) and ("ctx" in msg)


class _ForwardCaller:
    """Calls system.forward with an optional ctx, caching whether ctx is supported.

    This makes the ctx fallback explicit and easy to reason about:
    - Try with ctx once
    - If forward rejects ctx (unexpected keyword), permanently switch to calling without ctx
    - Do not swallow other TypeErrors
    """

    def __init__(self, forward_fn: Any):
        self._forward = forward_fn
        self._supports_ctx: bool | None = None  # None = unknown (probe), True/False = cached

    def __call__(self, batch_td: TensorDictBase, ctx: object | None) -> object:
        if self._supports_ctx is False:
            return self._forward(batch_td)
        if self._supports_ctx is True:
            return self._forward(batch_td, ctx=ctx)

        # Probe once.
        try:
            out = self._forward(batch_td, ctx=ctx)
            self._supports_ctx = True
            return out
        except TypeError as e:
            if _is_unexpected_ctx_kwarg_error(e):
                self._supports_ctx = False
                return self._forward(batch_td)
            raise


class _LayerStatsCollector:
    """Collect lightweight activation stats from attention layers via forward hooks."""

    def __init__(self, attn_modules: list[tuple[int, str, nn.Module]]) -> None:
        self._attn_modules = attn_modules
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.counts: dict[int, int] = {}
        self.sum_ms: dict[int, float] = {}
        self.sum_ma: dict[int, float] = {}
        self.max_abs: dict[int, float] = {}
        self.shapes: dict[int, list[int]] = {}

    def observe(self, idx: int, y: Tensor) -> None:
        # Collect lightweight stats; keep overhead bounded.
        try:
            z = y.detach().float()
            ms = float((z * z).mean().item())
            ma = float(z.abs().mean().item())
            mx = float(z.abs().max().item())
            self.counts[idx] = int(self.counts.get(idx, 0)) + 1
            self.sum_ms[idx] = float(self.sum_ms.get(idx, 0.0) + ms)
            self.sum_ma[idx] = float(self.sum_ma.get(idx, 0.0) + ma)
            self.max_abs[idx] = float(max(self.max_abs.get(idx, 0.0), mx))
            if idx not in self.shapes:
                self.shapes[idx] = [int(x) for x in list(z.shape)]
        except Exception as e:
            logger.warning(f"Failed to observe layer stats: {e}\n{traceback.format_exc()}")

    def finalize(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for idx, name, mod in self._attn_modules:
            n = int(self.counts.get(idx, 0))
            if n <= 0:
                continue
            ms = float(self.sum_ms.get(idx, 0.0)) / float(n)
            ma = float(self.sum_ma.get(idx, 0.0)) / float(n)
            mx = float(self.max_abs.get(idx, 0.0))
            rms = float(math.sqrt(max(0.0, ms)))
            cfg = getattr(mod, "config", None)
            out.append(
                {
                    "index": int(idx),
                    "name": str(name),
                    "type": "attention",
                    "shape": self.shapes.get(idx, None),
                    "mean_abs": float(ma),
                    "rms": float(rms),
                    "max_abs": float(mx),
                    "mode": str(
                        getattr(getattr(mod, "mode", None), "value", getattr(mod, "mode", ""))
                    ),
                    "null_attn": bool(getattr(cfg, "null_attn", False)) if cfg is not None else False,
                    "tie_qk": bool(getattr(cfg, "tie_qk", False)) if cfg is not None else False,
                    "rope_semantic": bool(getattr(cfg, "rope_semantic", False)) if cfg is not None else False,
                    "decoupled_gate": bool(getattr(cfg, "decoupled_gate", False)) if cfg is not None else False,
                }
            )
        return out


class _LayerStatsManager:
    def __init__(self, system: object, interval: int) -> None:
        self.interval = max(0, int(interval))
        self.enabled = self.interval > 0

        self.attn_modules: list[tuple[int, str, nn.Module]] = []
        self.mosaic_modules: list[tuple[int, str, nn.Module]] = []
        self._handles: list[RemovableHandle] = []
        self._collector = _LayerStatsCollector(self.attn_modules)

        if not self.enabled:
            return

        root_mod = getattr(system, "module", None)
        if not isinstance(root_mod, nn.Module):
            self.enabled = False
            return

        # Discover modules and attach stable viz IDs.
        for name, mod in root_mod.named_modules():
            if isinstance(mod, AttentionLayer):
                idx = int(len(self.attn_modules))
                mod._viz_index = int(idx)  # type: ignore[attr-defined]
                mod._viz_name = str(name)  # type: ignore[attr-defined]
                self.attn_modules.append((idx, str(name), mod))
            if isinstance(mod, MemoryBlockLayer):
                idx = int(len(self.mosaic_modules))
                mod._mosaic_index = int(idx)  # type: ignore[attr-defined]
                mod._mosaic_name = str(name)  # type: ignore[attr-defined]
                self.mosaic_modules.append((idx, str(name), mod))

        if not self.attn_modules:
            self.enabled = False
            return

        def _make_hook(i: int):
            def _hook(_m: nn.Module, _inp: tuple[object, ...], out: object) -> None:
                if not self._collector.enabled:
                    return
                y = (
                    out[0]
                    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], Tensor)
                    else out
                )
                if isinstance(y, Tensor):
                    self._collector.observe(i, y)

            return _hook

        for idx, name, mod in self.attn_modules:
            try:
                self._handles.append(mod.register_forward_hook(_make_hook(idx)))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register forward hook on attention layer {name!r} (index {idx}): {e}"
                ) from e

    def begin_step(self, step_1based: int) -> None:
        if not self.enabled:
            return
        self._collector.reset()
        self._collector.enabled = bool((step_1based % self.interval) == 0)

    def end_step(self) -> None:
        if not self.enabled:
            return
        self._collector.enabled = False

    def should_log(self, step_1based: int) -> bool:
        return bool(self.enabled and (step_1based % self.interval) == 0)

    def payload(self) -> dict[str, object]:
        return {"layers": self._collector.finalize()}

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception as e:
                raise RuntimeError("Failed to remove hook") from e
        self._handles.clear()


class StandardTrainer:
    def __init__(
        self,
        *,
        checkpoint_dir: str | None = None,
        runtime_plan_builder: RuntimePlanBuilder | None = None,
        train_dataloader_builder: TrainDataLoaderBuilder | None = None,
    ) -> None:
        self._checkpoint_dir_override = checkpoint_dir
        self._runtime_plan_builder = runtime_plan_builder or RuntimePlanBuilder()
        self._train_dataloader_builder = train_dataloader_builder or TrainDataLoaderBuilder()
        # Track loss from previous step for memory tuner metrics
        self._last_step_loss: float | None = None
        # Track accuracy from previous step for memory tuner metrics
        self._last_step_accuracy: float | None = None

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: _Engine,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            logger.info("Dry run requested, skipping training")
            return None

        data_spec = target.data
        if data_spec.ref == "dataset.tokens":
            cfg = dict(data_spec.config)
            if "tokenizer" not in cfg:
                cfg["tokenizer"] = str(getattr(manifest.defaults.data, "tokenizer", "tiktoken"))
            data_spec = data_spec.model_copy(update={"config": cfg})
        dataset_comp = engine.registry.build(data_spec, backend=str(target.backend))
        system = engine.registry.build(target.system, backend=str(target.backend))
        objective = engine.registry.build(target.objective, backend=str(target.backend))

        ckpt_dir = (
            Path(self._checkpoint_dir_override)
            if self._checkpoint_dir_override
            else Path("runs") / target.name
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        run_logger = RunLogger(ckpt_dir, filename="train.jsonl", enabled=True)

        last_device: torch.device | None = None
        for run in target.runs:
            if run.train is None:
                raise ValueError(f"Run {run.id} has no train config.")
            if run.train.phase != TrainPhase.STANDARD:
                raise ValueError(
                    f"trainer.standard only supports phase=standard, got {run.train.phase}"
                )

            self._run_single(
                defaults=manifest.defaults,
                target=target,
                run=run,
                train=run.train,
                dataset_comp=dataset_comp,
                system=system,
                objective=objective,
                checkpoint_dir=ckpt_dir,
                run_logger=run_logger,
            )
            last_device = torch.device(run.train.device)

        return {"system": system, "device": last_device, "checkpoint_dir": ckpt_dir}

    def _run_single(
        self,
        *,
        defaults: Defaults,
        target: ExperimentTargetConfig,
        run: Run,
        train: TrainConfig,
        dataset_comp: object,
        system: object,
        objective: object,
        checkpoint_dir: Path,
        run_logger: RunLogger,
    ) -> None:
        from cProfile import Profile
        from pstats import SortKey, Stats

        with Profile() as profile:
            torch.manual_seed(run.seed)
            device = torch.device(train.device)

            # Reset shared tuner for new training runs
            reset_shared_tuner()

            dist_ctx: DistributedContext | None = None
            dist_strategy = str(getattr(train, "distributed_strategy", "none")).lower()
            if dist_strategy != "none":
                try:
                    cfg = DistributedConfig(
                        strategy=DistributedStrategy(dist_strategy),
                        backend=str(getattr(train, "distributed_backend", "nccl")),
                    )
                    dist_ctx = DistributedContext.init(cfg)
                    device = dist_ctx.device
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize distributed training: {e}") from e

                # Only rank0 writes logs.
                try:
                    if hasattr(dist_ctx, "is_main") and not bool(getattr(dist_ctx, "is_main")):
                        run_logger.enabled = False
                except Exception as e:
                    raise RuntimeError("Failed to check if this is the main process") from e

            # W&B writer.
            wandb_writer: WandBWriter | None = None
            if bool(getattr(defaults.logging, "wandb", False)):
                is_main = True
                if dist_ctx is not None and hasattr(dist_ctx, "is_main"):
                    try:
                        is_main = bool(getattr(dist_ctx, "is_main"))
                    except Exception as e:
                        raise RuntimeError("Failed to check if this is the main process") from e

                if is_main:
                    try:
                        wandb_writer = WandBWriter(
                            out_dir=checkpoint_dir / "wandb" / str(run.id),
                            enabled=True,
                            project=str(getattr(defaults.logging, "wandb_project", "")),
                            entity=str(getattr(defaults.logging, "wandb_entity", "") or "") or None,
                            mode=str(getattr(defaults.logging, "wandb_mode", "online")),
                            run_name=f"{target.name}:{run.id}",
                            group=str(target.name),
                            tags=["standard"],
                            config={
                                "trainer": "standard",
                                "target": str(target.name),
                                "run_id": str(run.id),
                                "device": str(device),
                            },
                        )
                    except Exception as e:
                        wandb_writer = None
                        logger.fallback_warning(
                            "WARNING: W&B enabled but failed to initialize; continuing without W&B.\n"
                            f"reason={type(e).__name__}: {e}"
                        )

            runtime_plan = self._load_or_create_runtime_plan(
                checkpoint_dir=checkpoint_dir,
                device=device,
                train=train,
                system_cfg=dict(target.system.config),
            )
            dtype = self._parse_dtype(runtime_plan.dtype)

            if hasattr(system, "to"):
                system.to(device=device, dtype=dtype)  # type: ignore[attr-defined]

            # Explicit checkpoint loading (manifest-driven).
            load_ckpt = getattr(train, "load_checkpoint", None)
            if load_ckpt:
                logger.info(f"Loading explicit checkpoint: {load_ckpt}")
                try:
                    payload = torch.load(str(load_ckpt), map_location=device)
                    if isinstance(payload, dict) and "system_state_dict" in payload:
                        state = payload["system_state_dict"]
                    elif isinstance(payload, dict) and "model" in payload:
                        state = payload["model"]
                    else:
                        state = payload
                    
                    if not isinstance(state, dict):
                        raise TypeError(f"Checkpoint must be a dict, got {type(state).__name__}")

                    # Handle DDP wrapper
                    system_mod = getattr(system, "module", system)
                    if not isinstance(system_mod, nn.Module):
                        raise TypeError(f"System module is not an nn.Module, got {type(system_mod).__name__}")
                    missing, unexpected = system_mod.load_state_dict(state, strict=False)
                    if missing:
                        logger.warning(
                            "Explicit load missing keys: "
                            f"{len(missing)}\n{_format_state_keys_sample(list(missing))}"
                        )
                    if unexpected:
                        logger.warning(
                            "Explicit load unexpected keys: "
                            f"{len(unexpected)}\n{_format_state_keys_sample(list(unexpected))}"
                        )
                    if missing or unexpected:
                        # Persist the full lists for debugging (too noisy to log in full).
                        mismatch_path = Path(checkpoint_dir) / "explicit_load_key_mismatch.json"
                        mismatch_path.write_text(
                            json.dumps(
                                {
                                    "checkpoint": str(load_ckpt),
                                    "missing_keys": [str(x) for x in list(missing)],
                                    "unexpected_keys": [str(x) for x in list(unexpected)],
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    logger.success(f"Loaded weights from {load_ckpt}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load explicit checkpoint {load_ckpt}") from e

            # Optional UAA teacher: must be created before wrapping/compiling the student.
            uaa_state: UAAState | None = None
            uaa_cfg = getattr(train, "uaa", None)
            if uaa_cfg is not None and bool(getattr(uaa_cfg, "enabled", False)):
                if not _HAS_UAA_CTX:
                    logger.warning(
                        "UAA is enabled in the manifest, but this build does not provide TrainingUAAContext. "
                        "Disabling UAA for this run to avoid import/type issues."
                    )
                    uaa_cfg = None
                    uaa_state = None
                else:
                    system_any = cast(Any, system)
                    student_mod = getattr(system_any, "module", None)
                    if not isinstance(student_mod, nn.Module):
                        raise TypeError("UAA requires system.module to be an nn.Module.")

                    teacher_mode = str(getattr(uaa_cfg, "teacher", "init"))
                    if teacher_mode == "init":
                        # Avoid deepcopy() on nn.Module graphs with custom factory __new__/__init__
                        # (it can trigger constructor calls with missing args). Instead, build a
                        # fresh module of the same type/config and load weights.
                        cfg = getattr(student_mod, "config", None)
                        if cfg is None:
                            raise TypeError("UAA teacher=init requires student_mod.config to exist (ModelConfig).")
                        teacher_mod = type(student_mod)(cfg)  # type: ignore[call-arg]
                        teacher_mod.load_state_dict(student_mod.state_dict(), strict=True)
                    elif teacher_mode == "ckpt":
                        ckpt = getattr(train, "teacher_ckpt", None)
                        if not isinstance(ckpt, str) or not ckpt:
                            raise ValueError("UAA teacher=ckpt requires train.teacher_ckpt to be set.")
                        if str(ckpt).startswith("hf://"):
                            raise ValueError("UAA teacher=ckpt does not support hf:// checkpoints (use init for now).")
                        cfg = getattr(student_mod, "config", None)
                        if cfg is None:
                            raise TypeError("UAA teacher=ckpt requires student_mod.config to exist (ModelConfig).")
                        teacher_mod = type(student_mod)(cfg)  # type: ignore[call-arg]
                        payload = torch.load(str(ckpt), map_location="cpu")
                        if isinstance(payload, dict) and "system_state_dict" in payload:
                            state = payload["system_state_dict"]
                        else:
                            state = payload
                        if not isinstance(state, dict):
                            raise TypeError("UAA teacher checkpoint must be a state_dict or contain system_state_dict.")
                        missing, unexpected = teacher_mod.load_state_dict(state, strict=False)
                        if missing or unexpected:
                            logger.warning(
                                "UAA teacher checkpoint load had mismatches; continuing with strict=False.\n"
                                f"missing={len(missing)} unexpected={len(unexpected)}"
                            )
                    else:
                        raise ValueError(f"Unknown UAA teacher mode: {teacher_mode!r}")

                    teacher_mod = teacher_mod.to(device=device, dtype=dtype)
                    uaa_state = UAAState(teacher=teacher_mod, cfg=cast(Any, uaa_cfg))

            # CUDA perf knobs.
            #
            # TF32 is safe and commonly enabled on Ampere+ for faster fp32 matmuls
            # (many reductions / softmax stats run in fp32 even when weights are bf16).
            if device.type == "cuda":
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # PyTorch 2.x: nudge matmul towards faster kernels.
                    torch.set_float32_matmul_precision("high")
                except Exception as e:
                    logger.fallback_warning(
                    f"WARNING: Failed to enable TF32; continuing without TF32. device={device.type} dtype={dtype} error={e}"
                    )

            # Wrap system module for DDP/FSDP if requested.
            if dist_ctx is not None:
                try:
                    # `system` is intentionally generic (typed as `object`), but some systems
                    # expose a `.module: nn.Module` attribute that we can wrap/replace.
                    # Cast to Any so Pyright/Pylance allows dynamic attribute assignment.
                    system_any = cast(Any, system)
                    m = getattr(system_any, "module", None)
                    if isinstance(m, nn.Module):
                        system_any.module = dist_ctx.wrap_model(m)
                except Exception as e:
                    raise RuntimeError("Failed to wrap system module for DDP/FSDP") from e

            # torch.compile status (set later; must be defined for telemetry).
            compiled = False
            compiled_orig_module: nn.Module | None = None

            loader = self._build_loader(
                dataset_comp=dataset_comp,
                defaults=defaults,
                train=train,
                device=device,
                batch_size=int(runtime_plan.batch_size),
                dist_ctx=dist_ctx,
            )

            # Export reproducibility artifacts.
            #
            # IMPORTANT: export IO shapes BEFORE torch.compile.
            # Some Inductor versions can fail during the forward-only probe used for IO export.
            self._export_compiled_plan(checkpoint_dir=checkpoint_dir, target=target)
            self._export_io_shapes(
                checkpoint_dir=checkpoint_dir,
                loader=loader,
                system=system,
                device=device,
            )

            # Activation checkpointing: enable on topology modules when requested.
            #
            # The topology stack supports checkpointing, but we must flip the flags here
            # based on `TrainConfig` (otherwise the knobs are silently ignored).
            if bool(getattr(train, "activation_checkpointing", False)):
                thr_mb = float(getattr(train, "activation_checkpoint_threshold_mb", 0.0) or 0.0)
                try:
                    system_any = cast(Any, system)
                    root = getattr(system_any, "module", None)
                    if isinstance(root, nn.Module):
                        self._enable_activation_checkpointing(root, enabled=True, threshold_mb=thr_mb)
                    elif isinstance(system, nn.Module):
                        self._enable_activation_checkpointing(system, enabled=True, threshold_mb=thr_mb)
                except Exception as e:
                    raise RuntimeError("Failed to enable activation checkpointing") from e

            # Optional torch.compile (if requested and it fails, raise).
            #
            # NOTE: torch.compile's "reduce-overhead" mode uses CUDA graphs. When we invoke the
            # model multiple times per optimizer step (gradient accumulation), CUDA-graph output
            # buffer reuse can trigger correctness errors:
            #   "accessing tensor output of CUDAGraphs that has been overwritten..."
            # In that regime, we must avoid CUDA graphs.
            if bool(getattr(runtime_plan, "compile", False)):
                try:
                    system_any = cast(Any, system)
                    m = getattr(system_any, "module", None)
                    if isinstance(m, nn.Module):
                        compiled_orig_module = m
                        compile_mode = str(getattr(runtime_plan, "compile_mode", "default"))

                        # Suppress confusing CUDA-specific warnings on MPS
                        if device.type == "mps":
                            import warnings
                            warnings.filterwarnings(
                                "ignore",
                                message=".*Not enough SMs to use max_autotune_gemm mode.*",
                                category=UserWarning
                            )
                            if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
                                logger.info(
                                    f"torch.compile mode '{compile_mode}' on MPS - expect CUDA-optimized warnings "
                                    "to be suppressed for cleaner logging."
                                )
                        if device.type == "cuda":
                            accum_steps_cfg = max(1, int(getattr(train, "gradient_accumulation_steps", 1) or 1))
                            if compile_mode == "reduce-overhead" and int(accum_steps_cfg) > 1:
                                compile_mode = "default"
                                logger.info(
                                    "torch.compile mode reduced from 'reduce-overhead' to 'default' because "
                                    f"gradient_accumulation_steps={accum_steps_cfg} can trip CUDA-graph output reuse."
                                )
                            # max-autotune can also enable CUDA graphs internally; prefer the no-cudagraphs
                            # variant when we have multiple invocations per step.
                            if compile_mode == "max-autotune" and int(accum_steps_cfg) > 1:
                                compile_mode = "max-autotune-no-cudagraphs"
                                logger.info(
                                    "torch.compile mode adjusted from 'max-autotune' to 'max-autotune-no-cudagraphs' because "
                                    f"gradient_accumulation_steps={accum_steps_cfg} requires avoiding CUDA-graph output reuse."
                                )
                        try:
                            logger.info(f"Compiling model with torch.compile (mode={compile_mode}) - this may take 30-60 seconds...")
                            # Use static-shape compilation when possible.
                            #
                            # The Dynamo/Inductor stack can hit internal SymPy errors on some builds when
                            # dynamic-shape machinery is enabled (e.g. AttributeError: 'int' object has no
                            # attribute 'xreplace'). Our training shapes are intended to be static
                            # (fixed block_size), so prefer dynamic=False when supported.
                            try:
                                # Extra safety: disable Dynamo dynamic-shape machinery globally if available.
                                # This avoids SymPy-based shape substitutions that can crash with `.xreplace`
                                # on some builds, while matching our fixed block_size training regime.
                                try:
                                    import torch._dynamo as _dynamo  # type: ignore

                                    if hasattr(_dynamo, "config") and hasattr(_dynamo.config, "dynamic_shapes"):
                                        _dynamo.config.dynamic_shapes = False  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                system_any.module = torch.compile(m, mode=str(compile_mode), dynamic=False)
                            except TypeError:
                                system_any.module = torch.compile(m, mode=str(compile_mode))
                        except Exception as e:
                            # If the runtime doesn't recognize a mode (older PyTorch), fall back to default.
                            if str(compile_mode) == "max-autotune-no-cudagraphs":
                                logger.warning(
                                    "torch.compile mode 'max-autotune-no-cudagraphs' failed; falling back to 'default'. "
                                    f"Error: {type(e).__name__}: {e}"
                                )
                                try:
                                    system_any.module = torch.compile(m, mode="default", dynamic=False)
                                except TypeError:
                                    system_any.module = torch.compile(m, mode="default")
                            else:
                                raise
                        compiled = True
                        if device.type == "mps":
                            logger.success(
                                f"torch.compile enabled (mode={compile_mode}) for {target.name}:{run.id} - "
                                "optimized for Apple Silicon MPS backend"
                            )
                        else:
                            logger.success(
                                f"torch.compile enabled (mode={compile_mode}) for {target.name}:{run.id}"
                            )
                    else:
                        logger.info(
                            f"torch.compile requested but no nn.Module found on system.module "
                            f"(target={target.name} run={run.id}); continuing without compile."
                        )
                except Exception as e:
                    raise RuntimeError("Failed to compile model") from e
            stats = Stats(profile)
            stats.strip_dirs()
            stats.sort_stats(SortKey.TIME)
            stats.print_stats()
            stats.reverse_order()

        # Helper: if we hit the known Dynamo/SymPy `.xreplace` crash during the first compiled
        # execution, fall back to eager (so the run continues).
        def _maybe_fallback_from_xreplace(exc: BaseException) -> bool:
            nonlocal compiled
            if (not compiled) or compiled_orig_module is None:
                return False
            msg = f"{type(exc).__name__}: {exc}"
            if "xreplace" not in msg:
                return False
            try:
                system_any = cast(Any, system)
                system_any.module = compiled_orig_module
                compiled = False
                logger.fallback_warning(
                    "WARNING: torch.compile hit internal Dynamo/SymPy error (xreplace). "
                    "Falling back to eager execution to keep the run alive."
                )
                return True
            except Exception:
                return False

        optimizer = self._build_optimizer(
            system=system,
            train=train,
            device=device,
            dtype=dtype,
        )

        lr_scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        swap = SwapManager(
            offload_optimizer=bool(getattr(train, "offload_optimizer", False)),
            offload_device="cpu",
        )

        use_amp = bool(train.use_amp)
        amp_dtype = autocast_dtype(device, str(train.amp_dtype))

        telemetry_interval = int(getattr(train, "telemetry_interval", 10)) or 10
        telemetry_interval = max(1, telemetry_interval)

        profile_every = int(getattr(train, "profile_every", 0)) or 0
        profile_every = max(0, profile_every)
        profile_record_shapes = bool(getattr(train, "profile_record_shapes", False))

        layer_stats = _LayerStatsManager(
            system=system,
            interval=int(getattr(train, "layer_telemetry_interval", 0) or 0),
        )

        # Precompute static-ish memory numbers once.
        param_mb = _bytes_to_mb(self._param_bytes(system))

        # Emit run metadata once.
        try:
            run_logger.log_event(
                type="telemetry",
                run_id=str(run.id),
                phase="standard",
                step=0,
                data={
                    "device": str(device),
                    "dtype": str(dtype),
                    "amp": bool(use_amp),
                    "amp_dtype": str(amp_dtype),
                    "compiled": bool(compiled),
                    "compile_mode": str(getattr(runtime_plan, "compile_mode", "")),
                    "param_mb": float(param_mb),
                },
            )
        except Exception as e:
            raise RuntimeError("Failed to emit telemetry") from e

        logger.header("Training", f"{target.name}:{run.id} • {run.steps} steps")

        if not hasattr(objective, "loss"):
            raise TypeError("Objective component does not expose loss()")
        loss_fn = objective.loss  # type: ignore[attr-defined]
        loss_batch_key = self._resolve_loss_batch_key(loss_fn)

        metrics_fn = getattr(objective, "metrics", None)
        metrics_sig = None
        metrics_params: dict[str, inspect.Parameter] | None = None
        metrics_accepts_kwargs = False
        if callable(metrics_fn):
            try:
                metrics_sig = inspect.signature(metrics_fn)
                metrics_params = dict(metrics_sig.parameters)
                metrics_accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in metrics_params.values()
                )
            except Exception as e:
                raise RuntimeError("Failed to introspect objective metrics function") from e

        def call_objective_loss(*, outputs: object, batch_td: TensorDictBase) -> Tensor:
            if loss_batch_key == "batch":
                loss = loss_fn(outputs=outputs, batch=batch_td)
            elif loss_batch_key == "_batch":
                loss = loss_fn(outputs=outputs, _batch=batch_td)
            else:
                loss = loss_fn(outputs=outputs, batch_td=batch_td)

            if not isinstance(loss, Tensor):
                raise TypeError(f"Objective.loss must return a Tensor, got {type(loss).__name__}")
            return loss

        def call_objective_metrics(
            *, outputs: object, batch_td: TensorDictBase, loss: Tensor
        ) -> dict[str, float] | None:
            if not callable(metrics_fn) or metrics_params is None:
                return None

            kwargs: dict[str, object] = {}
            if "outputs" in metrics_params or metrics_accepts_kwargs:
                kwargs["outputs"] = outputs

            if "batch" in metrics_params:
                kwargs["batch"] = batch_td
            elif "_batch" in metrics_params:
                kwargs["_batch"] = batch_td
            elif "batch_td" in metrics_params:
                kwargs["batch_td"] = batch_td
            elif metrics_accepts_kwargs:
                kwargs["batch"] = batch_td

            if "loss" in metrics_params:
                kwargs["loss"] = loss
            elif "_loss" in metrics_params:
                kwargs["_loss"] = loss
            elif metrics_accepts_kwargs:
                kwargs["loss"] = loss

            extra = metrics_fn(**kwargs)  # type: ignore[misc]
            return cast(dict[str, float] | None, extra) if isinstance(extra, dict) else None

        if not hasattr(system, "forward"):
            raise TypeError("System component does not expose forward()")
        forward_caller = _ForwardCaller(system.forward)  # type: ignore[attr-defined]

        table2 = Table2Telemetry()
        table2_writer = Table2SummaryWriter()

        last_table2_metrics: dict[str, float] | None = None
        loss_val_live: float | None = None

        loader_iter = iter(loader)

        try:
            with logger.progress_bar() as progress:
                task = progress.add_task("Training...", total=int(run.steps))

                for step0 in range(int(run.steps)):
                    step_1 = int(step0 + 1)
                    t_step0 = time.perf_counter()

                    # Viz context setup.
                    viz_interval = max(0, int(getattr(train, "viz_interval", 0) or 0))
                    viz_enabled = bool(viz_interval > 0 and (step_1 % viz_interval) == 0)

                    if uaa_state is not None and uaa_cfg is not None and bool(getattr(uaa_cfg, "enabled", False)) and _HAS_UAA_CTX:
                        # Default to a single layer/head unless explicitly specified (avoid accidental "capture all").
                        uaa_layers = tuple(int(x) for x in (getattr(uaa_cfg, "layers", None) or []) if int(x) >= 0)
                        uaa_heads = tuple(int(x) for x in (getattr(uaa_cfg, "heads", None) or []) if int(x) >= 0)
                        if not uaa_layers:
                            uaa_layers = (0,)
                        if not uaa_heads:
                            uaa_heads = (0,)
                        every = max(1, int(getattr(uaa_cfg, "every_steps", 1) or 1))
                        uaa_enabled_step = bool((step_1 % every) == 0)

                        # _TrainingUAAContext is only available when _HAS_UAA_CTX is True.
                        viz_ctx = cast(Any, _TrainingUAAContext)(
                            enabled=bool(viz_enabled),
                            step=int(step_1),
                            max_tokens=int(getattr(train, "viz_tokens", 16) or 16),
                            max_channels=int(getattr(train, "viz_channels", 32) or 32),
                            max_heads=int(getattr(train, "viz_heads", 4) or 4),
                            topk=int(getattr(train, "viz_topk", 8) or 8),
                            uaa_enabled=bool(uaa_enabled_step),
                            uaa_layers=uaa_layers,
                            uaa_heads=uaa_heads,
                            uaa_query_index=-1,
                        )
                    else:
                        viz_ctx = TrainingVizMosaicContext(
                            enabled=bool(viz_enabled),
                            step=int(step_1),
                            max_tokens=int(getattr(train, "viz_tokens", 16) or 16),
                            max_channels=int(getattr(train, "viz_channels", 32) or 32),
                            max_heads=int(getattr(train, "viz_heads", 4) or 4),
                            topk=int(getattr(train, "viz_topk", 8) or 8),
                        )

                    # Initialize with loss from previous step for memory tuner
                    viz_ctx._last_loss = self._last_step_loss
                    # Initialize with accuracy from previous step for memory tuner
                    viz_ctx.train_accuracy = self._last_step_accuracy

                    viz_ctx.memblock_teacher_p = self._compute_memblock_teacher_p(
                        train=train, step_1=step_1, total_steps=int(run.steps)
                    )

                    viz_ctx.memblock_stats_enabled = bool((step_1 % telemetry_interval) == 0)

                    try:
                        viz_ctx.memblock_write_warmup_steps = int(
                            getattr(train, "memblock_write_warmup_steps", 0) or 0
                        )
                    except Exception as e:
                        raise RuntimeError("Failed to set mosaic write warmup steps") from e

                    # Gradient accumulation.
                    accum_steps = max(1, int(getattr(train, "gradient_accumulation_steps", 1) or 1))

                    optimizer.zero_grad(set_to_none=True)

                    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                    outputs_last: object | None = None
                    loss_last: Tensor | None = None
                    last_batch_td: TensorDictBase | None = None
                    kernel_events_estimate: int | None = None

                    layer_stats.begin_step(step_1)

                    # Optional profiling (if enabled and it fails, raise).
                    did_profile_first = bool(profile_every > 0 and (step_1 % profile_every) == 0)
                    # Stream microbatches: avoid staging the full accumulation window on GPU.
                    # This reduces peak VRAM and host overhead, and can enable larger microbatches
                    # (a real throughput win) without activation checkpoint recompute.
                    t_data0 = time.perf_counter()
                    t_fwb0 = time.perf_counter()
                    did_profiled = False
                    prof = None
                    for mb_i in range(int(accum_steps)):
                        item, loader_iter = self._next_loader_item(loader, loader_iter)
                        if isinstance(item, TensorDictBase):
                            batch_td = item
                        elif isinstance(item, dict):
                            batch_td = as_tensordict(item)
                        else:
                            raise TypeError(
                                "StandardTrainer expects batch items to be dict/TensorDict. "
                                f"Got {type(item).__name__}."
                            )
                        mb = cast(
                            TensorDictBase,
                            to_device(
                                batch_td,
                                device=device,
                                non_blocking=bool(getattr(train, "pin_memory", False) and device.type == "cuda"),
                            ),
                        )

                        # torch.compile can use CUDA graphs that reuse output buffers across
                        # invocations. During gradient accumulation we invoke the model multiple
                        # times per optimizer step and may keep references to outputs (e.g.
                        # outputs_last for telemetry). Marking step boundaries prevents
                        # "accessing tensor output of CUDAGraphs that has been overwritten".
                        if compiled and device.type == "cuda":
                            try:
                                torch.compiler.cudagraph_mark_step_begin()
                            except Exception as e:
                                raise RuntimeError(
                                    "torch.compile is enabled on CUDA, but failed to mark cudagraph step boundary. "
                                    "This is required to safely access outputs across multiple model invocations "
                                    "within a single optimizer step (e.g. gradient accumulation)."
                                ) from e

                        # Profile exactly one microbatch (first in the accumulation) when enabled.
                        if did_profile_first and (not did_profiled):
                            try:
                                acts = [ProfilerActivity.CPU]
                                if device.type == "cuda":
                                    acts.append(ProfilerActivity.CUDA)

                                with profile(
                                    activities=acts,
                                    record_shapes=bool(profile_record_shapes),
                                    profile_memory=True,
                                ) as prof_ctx:
                                    outputs_last, loss_last = self._forward_loss(
                                        batch_td=mb,
                                        device=device,
                                        use_amp=use_amp,
                                        amp_dtype=amp_dtype,
                                        viz_ctx=viz_ctx,
                                        forward_caller=forward_caller,
                                        objective_loss=call_objective_loss,
                                        uaa_state=uaa_state,
                                    )
                                    loss_sum += loss_last.detach().float()
                                    (loss_last / float(accum_steps)).backward()
                                prof = prof_ctx
                                did_profiled = True
                            except Exception as e:
                                if _maybe_fallback_from_xreplace(e):
                                    # Retry once in eager mode.
                                    outputs_last, loss_last = self._forward_loss(
                                        batch_td=mb,
                                        device=device,
                                        use_amp=use_amp,
                                        amp_dtype=amp_dtype,
                                        viz_ctx=viz_ctx,
                                        forward_caller=forward_caller,
                                        objective_loss=call_objective_loss,
                                        uaa_state=uaa_state,
                                    )
                                    loss_sum += loss_last.detach().float()
                                    (loss_last / float(accum_steps)).backward()
                                    prof = None
                                    did_profiled = True
                                else:
                                    raise RuntimeError("Failed to profile") from e
                        else:
                            try:
                                outputs_last, loss_last = self._forward_loss(
                                    batch_td=mb,
                                    device=device,
                                    use_amp=use_amp,
                                    amp_dtype=amp_dtype,
                                    viz_ctx=viz_ctx,
                                    forward_caller=forward_caller,
                                    objective_loss=call_objective_loss,
                                    uaa_state=uaa_state,
                                )
                                loss_sum += loss_last.detach().float()
                                (loss_last / float(accum_steps)).backward()
                            except Exception as e:
                                if _maybe_fallback_from_xreplace(e):
                                    outputs_last, loss_last = self._forward_loss(
                                        batch_td=mb,
                                        device=device,
                                        use_amp=use_amp,
                                        amp_dtype=amp_dtype,
                                        viz_ctx=viz_ctx,
                                        forward_caller=forward_caller,
                                        objective_loss=call_objective_loss,
                                        uaa_state=uaa_state,
                                    )
                                    loss_sum += loss_last.detach().float()
                                    (loss_last / float(accum_steps)).backward()
                                else:
                                    raise

                        # Always track the last batch for metrics computation
                        last_batch_td = mb

                    t_data1 = time.perf_counter()
                    if prof is not None and device.type == "cuda":
                        try:
                            kernel_events_estimate = int(len(prof.events() or []))
                        except Exception as e:
                            raise RuntimeError("Failed to count profiler events") from e
                        # Persist a human-readable summary so we can identify the true bottleneck
                        # without guessing (e.g., attention vs MLP vs vocab projection/loss).
                        try:
                            prof_dir = checkpoint_dir / "profiles"
                            prof_dir.mkdir(parents=True, exist_ok=True)
                            prof_path = prof_dir / f"profile_step_{step_1}.txt"
                            table = prof.key_averages().table(
                                sort_by="cuda_time_total",
                                row_limit=40,
                            )
                            out = [str(table)]
                            if profile_record_shapes:
                                # Group by input shape so `aten::mm` gets split into its real callers.
                                out.append(
                                    prof.key_averages(group_by_input_shape=True).table(
                                        sort_by="cuda_time_total",
                                        row_limit=80,
                                    )
                                )
                            prof_path.write_text("\n\n".join(out) + "\n", encoding="utf-8")
                            logger.info(f"Saved torch profiler summary to {prof_path}")
                        except Exception as e:
                            raise RuntimeError("Failed to export torch profiler summary") from e

                    layer_stats.end_step()

                    t_fwb1 = time.perf_counter()

                    # Optional grad clipping.
                    clip = float(getattr(train, "grad_clip_norm", 0.0) or 0.0)
                    if clip > 0.0:
                        try:
                            torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                                system.parameters(),  # type: ignore[arg-type]
                                max_norm=float(clip),
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to clip gradients") from e

                    # Optimizer step.
                    try:
                        swap.before_optimizer_step(optimizer, device=device)
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        swap.after_optimizer_step(optimizer)
                        if bool(getattr(train, "offload_optimizer", False)):
                            optimizer.zero_grad(set_to_none=True)
                    except Exception as e:
                        raise RuntimeError("Optimizer step failed") from e

                    t_after_optim = time.perf_counter()

                    # Layer stats logging independent of main telemetry cadence.
                    if layer_stats.should_log(step_1):
                        try:
                            run_logger.log_event(
                                type="layer_stats",
                                run_id=str(run.id),
                                phase="standard",
                                step=int(step_1),
                                data=layer_stats.payload(),
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to log layer stats") from e

                    # Viz payload logging independent of main telemetry cadence.
                    if viz_ctx.enabled:
                        try:
                            run_logger.log_event(
                                type="viz",
                                run_id=str(run.id),
                                phase="standard",
                                step=int(step_1),
                                data=viz_ctx.to_event(),
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to log viz payload") from e

                    # Main telemetry.
                    if (step_1 % telemetry_interval) == 0:
                        metrics = self._build_step_metrics(
                            system=system,
                            train=train,
                            runtime_plan=runtime_plan,
                            optimizer=optimizer,
                            compiled=compiled,
                            step_time_s=float(max(1e-9, t_after_optim - t_step0)),
                            data_time_s=float(max(0.0, t_data1 - t_data0)),
                            fwd_bwd_time_s=float(max(0.0, t_fwb1 - t_fwb0)),
                            optim_time_s=float(max(0.0, t_after_optim - t_fwb1)),
                            accum_steps=accum_steps,
                            loss_sum=loss_sum,
                            loss_last=loss_last,
                            outputs_last=outputs_last,
                            last_batch_td=last_batch_td,
                            call_objective_metrics=call_objective_metrics,
                            table2=table2,
                            viz_ctx=viz_ctx,
                            param_mb=float(param_mb),
                            kernel_events_estimate=kernel_events_estimate,
                        )

                        loss_val_live = float(metrics.get("loss", metrics.get("train_loss", 0.0)))
                        last_table2_metrics = dict(metrics)

                        # Update persistent accuracy (loss is already updated after forward pass)
                        # Filter out -1 values which indicate bins with no samples
                        acc_vals = [v for k, v in metrics.items() if k.startswith("acc/bin_") and v >= 0]
                        if acc_vals:
                            self._last_step_accuracy = float(sum(acc_vals) / len(acc_vals))
                            # Update global metrics singleton with accuracy
                            update_training_metrics(step=step_1, accuracy=self._last_step_accuracy)

                        run_logger.log_metrics(
                            run_id=str(run.id),
                            phase="standard",
                            step=int(step_1),
                            metrics=metrics,
                        )

                        if wandb_writer is not None:
                            try:
                                wandb_writer.log_scalars(prefix="train", step=int(step_1), scalars=metrics)
                                wandb_writer.log_scalars(prefix="", step=int(step_1), scalars=metrics)
                            except Exception as e:
                                logger.fallback_warning(
                                    "WARNING: W&B logging failed for this step (continuing).\n"
                                    f"reason={type(e).__name__}: {e}"
                                )

                    # Progress always advances per optimizer step.
                    desc = f"Step {step_1}/{run.steps}"
                    if loss_val_live is not None:
                        desc = f"{desc} • loss={float(loss_val_live):.4f}"
                    progress.update(task, advance=1, description=desc)

        finally:
            # Cleanups must never depend on successful completion.
            try:
                if wandb_writer is not None:
                    wandb_writer.close()
            except Exception as e:
                logger.warning(f"Failed to close W&B writer (ignoring). error={e!r}")

            try:
                layer_stats.close()
            except Exception as e:
                raise RuntimeError("Failed to cleanup hooks") from e

        self._save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            run_id=str(run.id),
            phase="standard",
            step=int(run.steps),
            system=system,
        )

        # Table 2 export (only if Table 2 metrics were produced).
        if (
            bool(getattr(run_logger, "enabled", True))
            and last_table2_metrics is not None
            and any(k.startswith("acc/bin_") for k in last_table2_metrics.keys())
        ):
            self._export_table2_bundle(
                checkpoint_dir=checkpoint_dir,
                target=target,
                run=run,
                system=system,
                dataset_comp=dataset_comp,
                table2_writer=table2_writer,
                table2_cfg=table2.cfg,
                last_table2_metrics=last_table2_metrics,
            )

    def _forward_loss(
        self,
        *,
        batch_td: TensorDictBase,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        viz_ctx: TrainingVizMosaicContext,
        forward_caller: _ForwardCaller,
        objective_loss: Any,
        uaa_state: UAAState | None,
    ) -> tuple[object, Tensor]:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # Attach MOSAIC-friendly fields onto the ctx.
            inp = None
            try:
                inp = batch_td.get("input_ids", None)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError("Failed to get input ids") from e
            if isinstance(inp, Tensor):
                viz_ctx.input_ids = inp

            try:
                dl = batch_td.get("mosaic_drop_local", None)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError("Failed to get mosaic drop local") from e
            if isinstance(dl, Tensor):
                viz_ctx.mosaic_drop_local = dl

            teacher: dict[str, Tensor] = {}
            for k in ("read_bucket", "write_bucket", "write_gate", "clear"):
                try:
                    v = batch_td.get(f"memblock_teacher_{k}", None)  # type: ignore[attr-defined]
                except Exception as e:
                    raise RuntimeError("Failed to get mosaic teacher signal") from e
                if isinstance(v, Tensor):
                    teacher[k] = v
            viz_ctx.memblock_teacher = teacher or None

            # Decide whether to collect aux signals (objective-dependent).
            collect_aux = bool(teacher)
            try:
                if isinstance(getattr(objective_loss, "__self__", None), MosaicNextTokenWithAuxObjective):
                    collect_aux = True
            except Exception as e:
                raise RuntimeError("Failed to check if objective is MosaicNextTokenWithAuxObjective") from e

            viz_ctx.memblock_collect_aux = bool(collect_aux)

            # Speed: avoid threading a changing Python `ctx` object through the model when we
            # don't need it. Passing `ctx` can introduce graph breaks / retracing when
            # torch.compile is enabled, and can also trigger per-layer instrumentation checks.
            #
            # We only pass ctx when we are actually collecting aux signals or running viz.
            uaa_enabled = bool(getattr(viz_ctx, "uaa_enabled", False))
            ctx_to_pass: object | None = viz_ctx if (bool(viz_ctx.enabled) or bool(collect_aux) or uaa_enabled) else None

            outputs = forward_caller(batch_td, ctx_to_pass)

            # Merge MOSAIC aux outputs from ctx into outputs.
            try:
                aux_out = getattr(viz_ctx, "memblock_aux_out", None)
                if isinstance(outputs, dict) and isinstance(aux_out, dict):
                    for k, v in aux_out.items():
                        if (
                            isinstance(k, str)
                            and k.startswith("mosaic_")
                            and isinstance(v, Tensor)
                            and k not in outputs
                        ):
                            outputs[k] = v
            except Exception as e:
                raise RuntimeError("Failed to merge mosaic aux outputs") from e

            loss = objective_loss(outputs=outputs, batch_td=batch_td)

            # Optional UAA auxiliary loss (student-side; teacher is frozen).
            if (
                _HAS_UAA_CTX
                and uaa_state is not None
                and isinstance(outputs, dict)
                and (_TrainingUAAContext is not None)
                and isinstance(viz_ctx, _TrainingUAAContext)
            ):
                try:
                    uaa_loss = uaa_state.compute_loss(
                        ctx=cast(Any, viz_ctx),
                        batch=batch_td,
                        student_outputs=outputs,
                        step_1=int(viz_ctx.step),
                    )
                    if isinstance(uaa_loss, Tensor):
                        loss = loss + uaa_loss
                except Exception as e:
                    raise RuntimeError(f"Failed to compute UAA loss: {type(e).__name__}: {e}") from e

            if not isinstance(loss, Tensor):
                raise TypeError(f"Objective.loss must return a Tensor, got {type(loss).__name__}")

            # Store loss for next step's memory layer tuner
            self._last_step_loss = float(loss.detach())

            # Update global metrics singleton (zero overhead, no ctx needed)
            update_training_metrics(step=viz_ctx.step, loss=self._last_step_loss)

            return outputs, loss

    def _build_step_metrics(
        self,
        *,
        system: object,
        train: TrainConfig,
        runtime_plan: RuntimePlan,
        optimizer: torch.optim.Optimizer,
        compiled: bool,
        step_time_s: float,
        data_time_s: float,
        fwd_bwd_time_s: float,
        optim_time_s: float,
        accum_steps: int,
        loss_sum: Tensor,
        loss_last: Tensor | None,
        outputs_last: object | None,
        last_batch_td: TensorDictBase | None,
        call_objective_metrics: Any,
        table2: Table2Telemetry,
        viz_ctx: TrainingVizMosaicContext,
        param_mb: float,
        kernel_events_estimate: int | None,
    ) -> dict[str, float]:
        # Loss value (mean over microbatches).
        loss_val = float((loss_sum / float(accum_steps)).item())
        ppl = float(safe_perplexity_from_nll(float(loss_val)))

        lr = float(optimizer.param_groups[0].get("lr", float(train.lr)))
        lr_base = float(getattr(train, "lr", lr))
        lr_mult = (lr / lr_base) if lr_base > 0 else 1.0

        # Grad norm.
        grad_norm = 0.0
        try:
            grad_norm = float(global_grad_norm_l2(system))  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError("Failed to compute gradient norm") from e

        metrics: dict[str, float] = {
            "loss": float(loss_val),
            "train_loss": float(loss_val),
            "ppl": float(ppl),
            "train_ppl": float(ppl),
            "lr": float(lr),
            "lr_base": float(lr_base),
            "lr_mult": float(lr_mult),
            "grad_norm": float(grad_norm),
            "grad_accum": float(accum_steps),
            # Log the effective batch size (runtime plan can override train.batch_size).
            "batch_size": float(getattr(runtime_plan, "batch_size", getattr(train, "batch_size", 0))),
            "seq_len": float(getattr(train, "block_size", 0)),
            "compiled": 1.0 if compiled else 0.0,
            "time_step_s": float(step_time_s),
            "time_data_s": float(data_time_s),
            "time_fwd_bwd_s": float(fwd_bwd_time_s),
            "time_optim_s": float(optim_time_s),
            # Timing metrics (ms).
            "ms_step": float(step_time_s * 1000.0),
            "ms_data": float(data_time_s * 1000.0),
            "ms_fwd_bwd": float(fwd_bwd_time_s * 1000.0),
            "ms_opt": float(optim_time_s * 1000.0),
        }

        # Objective extra metrics.
        if outputs_last is not None and loss_last is not None and last_batch_td is not None:
            try:
                extra = call_objective_metrics(outputs=outputs_last, batch_td=last_batch_td, loss=loss_last)
                if isinstance(extra, dict):
                    metrics.update({str(k): float(v) for k, v in extra.items()})
            except Exception as e:
                raise RuntimeError("Failed to compute objective metrics") from e

        # UAA metrics (if present).
        if isinstance(outputs_last, dict):
            try:
                v = outputs_last.get("uaa/attn_kl", None)
                if isinstance(v, Tensor) and int(v.numel()) == 1:
                    metrics["uaa/attn_kl"] = float(v.detach().float().reshape(()))
            except Exception as e:
                raise RuntimeError("Failed to extract UAA metrics") from e

        # Table 2 telemetry.
        if last_batch_td is not None:
            try:
                has_table2_bin = isinstance(last_batch_td.get("table2_bin", None), Tensor)  # type: ignore[attr-defined]
                has_mem_teacher = (
                    isinstance(last_batch_td.get("memblock_teacher_read_bucket", None), Tensor)  # type: ignore[attr-defined]
                    and isinstance(last_batch_td.get("memblock_teacher_write_bucket", None), Tensor)  # type: ignore[attr-defined]
                    and isinstance(last_batch_td.get("memblock_teacher_write_gate", None), Tensor)  # type: ignore[attr-defined]
                )
                # Debug scalars: help diagnose "all -1" / "no reads" situations quickly in W&B.
                if has_table2_bin:
                    tb2 = last_batch_td.get("table2_bin", None)  # type: ignore[attr-defined]
                    if isinstance(tb2, Tensor):
                        valid = (tb2.detach() >= 0)
                        metrics["table2/valid_frac"] = float(valid.float().mean().item()) if tb2.numel() > 0 else 0.0
                        metrics["table2/valid_count"] = float(valid.float().sum().item())
                if has_mem_teacher:
                    rb2 = last_batch_td.get("memblock_teacher_read_bucket", None)  # type: ignore[attr-defined]
                    wb2 = last_batch_td.get("memblock_teacher_write_bucket", None)  # type: ignore[attr-defined]
                    wg2 = last_batch_td.get("memblock_teacher_write_gate", None)  # type: ignore[attr-defined]
                    if isinstance(rb2, Tensor) and rb2.numel() > 0:
                        metrics["mem/teacher_read_frac"] = float((rb2.detach() >= 0).float().mean().item())
                    if isinstance(wb2, Tensor) and wb2.numel() > 0:
                        metrics["mem/teacher_write_bucket_frac"] = float((wb2.detach() >= 0).float().mean().item())
                    if isinstance(wg2, Tensor) and wg2.numel() > 0:
                        metrics["mem/teacher_write_gate_mean"] = float(wg2.detach().float().mean().item())
                        metrics["mem/teacher_write_gate_fire_frac"] = float((wg2.detach().float() > 0.5).float().mean().item())
                if (has_table2_bin or has_mem_teacher) and (outputs_last is not None):
                    metrics.update(table2.compute(outputs=outputs_last, batch=last_batch_td))
            except Exception as e:
                raise RuntimeError("Failed to compute Table 2 telemetry") from e

        # Token throughput (for token-LM style datasets).
        if last_batch_td is not None:
            try:
                y = last_batch_td.get("target_ids", None)  # type: ignore[attr-defined]
                if isinstance(y, Tensor):
                    metrics["tok_s"] = float(y.numel() * int(accum_steps)) / float(max(1e-9, step_time_s))
            except Exception as e:
                raise RuntimeError("Failed to compute token throughput") from e

        # Memory footprint estimates (MiB).
        metrics["mem_params_mb"] = float(param_mb)
        try:
            metrics["mem_grads_mb"] = float(_bytes_to_mb(self._grad_bytes(system)))
        except Exception as e:
            logger.warning(f"Failed to estimate grad memory; continuing. error={e!r}")
        try:
            metrics["mem_optim_mb"] = float(_bytes_to_mb(self._optim_state_bytes(optimizer)))
        except Exception as e:
            logger.warning(f"Failed to estimate optimizer memory; continuing. error={e!r}")

        if kernel_events_estimate is not None:
            metrics["kernel_events_estimate"] = float(kernel_events_estimate)

        # MOSAIC memory stats.
        try:
            if isinstance(viz_ctx, TrainingVizMosaicContext) and viz_ctx.memblock_mem_stats:
                mosaic_f: dict[str, float] = {}
                t_keys: list[str] = []
                t_vals: list[Tensor] = []
                for k, v in viz_ctx.memblock_mem_stats.items():
                    kk = str(k)
                    if isinstance(v, (int, float)):
                        mosaic_f[kk] = float(v)
                    elif isinstance(v, Tensor) and v.numel() == 1:
                        t_keys.append(kk)
                        t_vals.append(v.detach().float())

                if t_vals:
                    stacked = torch.stack(t_vals)
                    flat = stacked.detach().cpu().tolist()
                    for kk, vv in zip(t_keys, flat, strict=True):
                        mosaic_f[kk] = float(vv)

                for kk, vv in mosaic_f.items():
                    metrics[kk] = float(vv)

                metrics["memblock_teacher_p"] = float(getattr(viz_ctx, "memblock_teacher_p", 1.0))

                def _avg(suffix: str) -> float | None:
                    vals = [float(v) for kk, v in mosaic_f.items() if str(kk).endswith(suffix)]
                    if not vals:
                        return None
                    return float(sum(vals) / float(len(vals)))

                # Table 2 stable namespaces.
                rg = _avg("/fuse_gate_mem_mean")
                wg = _avg("/write_gate_p_mean")
                re = _avg("/write_bucket_entropy_norm")
                if rg is not None:
                    metrics["mem/read_gate"] = float(rg)
                if wg is not None:
                    metrics["mem/write_gate"] = float(wg)
                if re is not None:
                    metrics["mem/routing_entropy"] = float(re)
        except Exception as e:
            raise RuntimeError("Failed to log MOSAIC memory stats") from e

        return metrics

    def _compute_memblock_teacher_p(self, *, train: TrainConfig, step_1: int, total_steps: int) -> float:
        try:
            warm = int(getattr(train, "memblock_teacher_p_warmup_steps", 0) or 0)
            cool = int(getattr(train, "memblock_teacher_p_cooldown_steps", 0) or 0)
            p0 = float(getattr(train, "memblock_teacher_p_start", 1.0))
            p1 = float(getattr(train, "memblock_teacher_p_end", 0.0))
            sched = str(getattr(train, "memblock_teacher_p_schedule", "linear")).lower()

            s_eff = max(0, min(int(total_steps), int(step_1)))
            denom = max(1, int(total_steps) - warm - cool)

            if s_eff <= warm:
                alpha = 0.0
            elif s_eff >= int(total_steps) - cool:
                alpha = 1.0
            else:
                alpha = float(s_eff - warm) / float(denom)

            if sched == "cosine":
                alpha2 = 0.5 - 0.5 * math.cos(math.pi * float(alpha))
            elif sched == "constant":
                alpha2 = 0.0
            else:
                alpha2 = float(alpha)

            p_t = float(p0 + (p1 - p0) * float(alpha2))
            return float(max(0.0, min(1.0, p_t)))
        except Exception as e:
            raise RuntimeError("Failed to compute mosaic teacher p") from e

    def _next_loader_item(
        self, loader: DataLoader, it: Iterable[Any]
    ) -> tuple[Any, Iterable[Any]]:
        """Cycles the dataloader instead of crashing when it is exhausted."""
        try:
            item = next(it)  # type: ignore[arg-type]
            return item, it
        except StopIteration:
            it2 = iter(loader)
            try:
                item = next(it2)
            except StopIteration as e:
                raise RuntimeError("Dataloader is empty; cannot fetch a batch") from e
            return item, it2

    def _build_optimizer(
        self,
        *,
        system: object,
        train: TrainConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.optim.Optimizer:
        if not hasattr(system, "parameters"):
            raise TypeError("System component does not expose parameters()")

        params = system.parameters()  # type: ignore[attr-defined]
        opt_name = str(getattr(train, "optimizer", "adamw")).lower()
        weight_decay = float(getattr(train, "weight_decay", 0.0))
        fused_opt = bool(getattr(train, "fused_optimizer", True))
        lr = float(train.lr)

        if opt_name in ("adamw", "adam"):
            if opt_name == "adamw":
                if fused_opt:
                    # CPU has no fused optimizer backend; make this explicit.
                    if device.type == "cpu":
                        logger.warning(
                            "AdamW fused optimizer requested on CPU; falling back to torch.optim.AdamW. "
                            "Set train.fused_optimizer=false to silence this warning."
                        )
                        fused_opt = False

                    use_master = (device.type == "mps" and dtype in (torch.float16, torch.float32)) or (
                        device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
                    )
                    if use_master:
                        logger.info(
                            f"optimizer=adamw fused=true backend=adamw_master device={device.type} dtype={dtype}"
                        )
                        return AdamWMaster(
                            params,  # type: ignore[arg-type]
                            lr=lr,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=float(weight_decay),
                            fused=True,
                        )
                    else:
                        # On accelerators, this is a performance-critical request; fail loud.
                        if device.type in ("cuda", "mps"):
                            raise RuntimeError(
                                "AdamW fused optimizer requested but unsupported for this device/dtype.\n"
                                f"device={device.type} dtype={dtype}\n"
                                "Supported: MPS fp16/fp32, CUDA fp16/bf16."
                            )

                logger.info(f"optimizer=adamw fused=false backend=torch_adamw device={device.type} dtype={dtype}")
                return torch.optim.AdamW(
                    params,  # type: ignore[arg-type]
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=float(weight_decay),
                )

            # opt_name == "adam"
            logger.info(f"optimizer=adam backend=torch_adam device={device.type} dtype={dtype}")
            return torch.optim.Adam(
                params,  # type: ignore[arg-type]
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(weight_decay),
            )

        if opt_name == "sgd":
            return torch.optim.SGD(
                params,  # type: ignore[arg-type]
                lr=lr,
                weight_decay=float(weight_decay),
            )

        if opt_name == "lion":
            return Lion(
                params,  # type: ignore[arg-type]
                lr=lr,
                weight_decay=float(weight_decay),
                fused=bool(fused_opt),
            )

        raise ValueError(f"Unknown optimizer {opt_name!r}")

    def _resolve_loss_batch_key(self, loss_fn: Any) -> str:
        try:
            loss_sig = inspect.signature(loss_fn)
            loss_params = loss_sig.parameters
        except Exception as e:
            raise RuntimeError("Failed to introspect objective loss function") from e

        if "outputs" not in loss_params:
            raise TypeError("Objective.loss must accept keyword argument 'outputs'")

        if "batch" in loss_params:
            return "batch"
        if "_batch" in loss_params:
            return "_batch"
        if "batch_td" in loss_params:
            return "batch_td"

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in loss_params.values()):
            return "batch"

        raise TypeError("Objective.loss must accept a batch keyword (e.g. 'batch' or '_batch')")

    def _export_compiled_plan(self, *, checkpoint_dir: Path, target: ExperimentTargetConfig) -> None:
        try:
            plan_txt = Planner().format_target(target, indent=0, path=f"targets[{target.name}]")
            (checkpoint_dir / "compiled_plan.txt").write_text("\n".join(plan_txt) + "\n", encoding="utf-8")
        except Exception as e:
            raise RuntimeError("Failed to export compiled plan (compiled_plan.txt)") from e

    def _export_io_shapes(
        self,
        *,
        checkpoint_dir: Path,
        loader: DataLoader,
        system: object,
        device: torch.device,
    ) -> None:
        try:
            it = iter(loader)
            b0 = next(it)
            b0 = b0 if isinstance(b0, TensorDictBase) else as_tensordict(b0)  # type: ignore[arg-type]
            b0 = cast(TensorDictBase, to_device(b0, device=device))

            try:
                b_export = b0[:1]
            except Exception as e:
                raise RuntimeError("Failed to slice a small batch for IO export") from e

            if not hasattr(system, "forward"):
                raise TypeError(
                    "Failed to export IO shapes: system has no forward().\n"
                    "Fix: ensure the system object exposes forward(batch, ...) so we can capture output shapes."
                )

            fwd = system.forward  # type: ignore[attr-defined]
            try:
                sig = inspect.signature(fwd)
            except Exception as e:
                raise RuntimeError("Failed to introspect system.forward signature for IO export") from e

            params = sig.parameters
            accepts_ctx = ("ctx" in params) or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )

            with torch.no_grad():
                if accepts_ctx:
                    o0 = fwd(b_export, ctx=None)  # type: ignore[call-arg]
                else:
                    o0 = fwd(b_export)  # type: ignore[call-arg]

            def shape_sig(td: object) -> dict[str, object]:
                out: dict[str, object] = {}
                if isinstance(td, Tensor):
                    out["__tensor__"] = {
                        "shape": list(td.shape),
                        "dtype": str(td.dtype),
                        "device": str(td.device),
                    }
                    return out
                if isinstance(td, dict):
                    items = td.items()
                else:
                    items = dict(td).items()  # type: ignore[arg-type]

                for k, v in items:
                    if isinstance(v, Tensor):
                        out[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                return out

            (checkpoint_dir / "io_shapes.json").write_text(
                json.dumps({"batch": shape_sig(b_export), "outputs": shape_sig(o0)}, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to export IO shapes.\n"
                "Why this matters: IO shape export is required for reproducibility artifacts.\n"
                "Fix:\n"
                "  - Ensure the system forward returns a Tensor or a dict-like of Tensors.\n"
                "  - Ensure system.forward accepts `ctx` or does not require it.\n"
                "  - Ensure the batch is dict/TensorDict-like and convertible via as_tensordict().\n"
                f"system={type(system).__name__}\n"
                f"error={type(e).__name__}: {e}\n"
            ) from e

    def _export_table2_bundle(
        self,
        *,
        checkpoint_dir: Path,
        target: ExperimentTargetConfig,
        run: Run,
        system: object,
        dataset_comp: object,
        table2_writer: Table2SummaryWriter,
        table2_cfg: Any,
        last_table2_metrics: dict[str, float],
    ) -> None:
        # Prefer extracting memory config from the model (for MOSAIC). Fall back to dataset component when needed.
        mb: int | None = None
        mh: int | None = None
        mod = getattr(system, "module", None)
        if isinstance(mod, nn.Module):
            buckets: set[int] = set()
            hashes: set[int] = set()
            for m in mod.modules():
                if isinstance(m, MemoryBlockLayer):
                    buckets.add(int(m.memory.mem_buckets))
                    hashes.add(int(m.memory.mem_hashes))
            if buckets and hashes:
                if len(buckets) != 1 or len(hashes) != 1:
                    raise ValueError("Inconsistent mem_buckets/mem_hashes across MemoryBlockLayer modules.")
                mb = next(iter(buckets))
                mh = next(iter(hashes))

        if mb is None or mh is None:
            mb2 = getattr(dataset_comp, "mem_buckets", None)
            mh2 = getattr(dataset_comp, "mem_hashes", None)
            if isinstance(mb2, int) and isinstance(mh2, int):
                mb = int(mb2)
                mh = int(mh2)

        if mb is None or mh is None:
            raise TypeError("Table 2 export requires mem_buckets/mem_hashes (from model or dataset).")

        params_fn = getattr(system, "parameters", None)
        if not callable(params_fn):
            raise TypeError("Table 2 export requires system.parameters().")

        n_params = 0
        params_iter = params_fn()
        if not isinstance(params_iter, Iterable):
            raise TypeError("system.parameters() must return an iterable")
        for p in params_iter:
            if not isinstance(p, nn.Parameter):
                raise TypeError("system.parameters() must yield nn.Parameter objects")
            n_params += int(p.numel())

        out_path = table2_writer.write(
            out_dir=checkpoint_dir,
            run_id=str(run.id),
            mem_buckets=int(mb),
            mem_hashes=int(mh),
            model_size=f"params={n_params}",
            metrics=last_table2_metrics,
            n_bins=int(table2_cfg.n_bins),
        )

        # Console UX: print both the path and a small table summary.
        logger.subheader("Table 2 export")
        logger.path(str(out_path), label="summary_json")
        logger.info(f"Table 2 summary written to [path]{out_path}[/path]")

        rows: list[list[str]] = []
        rows.append(["mem_buckets", str(int(mb))])
        rows.append(["mem_hashes", str(int(mh))])
        rows.append(["model_size", f"params={n_params}"])

        wb = float(last_table2_metrics.get("acc/worst_bin", -1.0))
        cr = float(last_table2_metrics.get("collision/wrong_item_read_rate", -1.0))
        rows.append(["acc/worst_bin", "—" if wb < 0.0 else f"{wb:.4f}"])
        rows.append(["collision/wrong_item_read_rate", "—" if cr < 0.0 else f"{cr:.4f}"])

        for i in range(int(table2_cfg.n_bins)):
            k = f"acc/bin_{i}"
            v = float(last_table2_metrics.get(k, -1.0))
            rows.append([k, "—" if v < 0.0 else f"{v:.4f}"])

        logger.table(
            title=f"Table 2 summary • {target.name}:{run.id}",
            columns=["Metric", "Value"],
            rows=rows,
        )

    def _build_loader(
        self,
        *,
        dataset_comp: object,
        defaults: Defaults,
        train: TrainConfig,
        device: torch.device,
        batch_size: int,
        dist_ctx: object | None = None,
    ) -> DataLoader:
        return self._train_dataloader_builder.build(
            dataset_comp=dataset_comp,
            defaults=defaults,
            train=train,
            device=device,
            batch_size=int(batch_size),
            dist_ctx=dist_ctx,
        )

    def _save_checkpoint(
        self,
        *,
        checkpoint_dir: Path,
        run_id: str,
        phase: str,
        step: int,
        system: object,
    ) -> Path:
        filename = f"{run_id}_{phase}_final.pt"
        path = checkpoint_dir / filename
        if not hasattr(system, "state_dict"):
            raise TypeError("System component does not expose state_dict()")
        state = {
            "system_state_dict": system.state_dict(),  # type: ignore[attr-defined]
            "run_id": run_id,
            "step": step,
        }
        torch.save(state, path)
        return path

    def _parse_dtype(self, dtype: str) -> torch.dtype:
        dt = dtype.lower()
        if dt == "float32":
            return torch.float32
        if dt == "float16":
            return torch.float16
        if dt == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _enable_activation_checkpointing(self, root: nn.Module, *, enabled: bool, threshold_mb: float) -> None:
        """Enable activation checkpointing on supported topology modules."""
        for m in root.modules():
            if hasattr(m, "activation_checkpointing"):
                try:
                    setattr(m, "activation_checkpointing", bool(enabled))
                except Exception as e:
                    logger.warning(
                        f"Failed to set activation_checkpointing on {type(m).__name__}; continuing. error={e!r}"
                    )
            if hasattr(m, "activation_checkpoint_threshold_mb"):
                try:
                    setattr(m, "activation_checkpoint_threshold_mb", float(threshold_mb))
                except Exception as e:
                    logger.warning(
                        f"Failed to set activation_checkpoint_threshold_mb on {type(m).__name__}; continuing. error={e!r}"
                    )

    def _load_or_create_runtime_plan(
        self,
        *,
        checkpoint_dir: Path,
        device: torch.device,
        train: TrainConfig,
        system_cfg: dict[str, Any],
    ) -> RuntimePlan:
        train_payload = train.model_dump()
        train_payload.pop("teacher_ckpt", None)
        payload: dict[str, Any] = {
            "device": str(device),
            "torch": torch.__version__,
            "system": system_cfg,
            "train": train_payload,
        }
        return self._runtime_plan_builder.build(
            checkpoint_dir=checkpoint_dir,
            device=device,
            train=train,
            payload=payload,
        )

    def _param_bytes(self, system: object) -> int:
        total = 0
        for p in system.parameters():  # type: ignore[attr-defined]
            if isinstance(p, Tensor):
                total += int(p.numel()) * int(p.element_size())
        return total

    def _grad_bytes(self, system: object) -> int:
        total = 0
        for p in system.parameters():  # type: ignore[attr-defined]
            g = getattr(p, "grad", None)
            if isinstance(g, Tensor):
                recognized = int(g.numel()) * int(g.element_size())
                total += recognized
        return total

    def _optim_state_bytes(self, optimizer: torch.optim.Optimizer) -> int:
        total = 0
        for _param, state in optimizer.state.items():
            if isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, Tensor):
                        total += int(v.numel()) * int(v.element_size())
        return total
