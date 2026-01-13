"""Standard (non-upcycling) training loop implementation.

This module intentionally contains the "big loop" so `trainer/trainer.py` can
stay as a thin, composable orchestrator.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from caramba.carmath import autocast_dtype
from caramba.collector.layer.hook import LayerStatsHook
from caramba.collector.training.checkpoint import FinalCheckpointHook
from caramba.collector.training.logging import RunLoggerHook, WandBHook
from caramba.collector.training.metrics import build_standard_step_metrics
from caramba.collector.training.table2 import Table2ExportHook
from caramba.collector.training.viz import VizHook
from caramba.compiler.plan import Planner
from caramba.config.defaults import Defaults
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig
from caramba.console import logger
from caramba.instrumentation import RunLogger
from caramba.instrumentation.training_metrics import update_training_metrics
from caramba.instrumentation.viz import TrainingVizMosaicContext
from caramba.instrumentation.wandb_writer import WandBWriter
from caramba.layer.memory_block.memory.tuner import reset_shared_tuner
from caramba.optimizer.builder import build_optimizer
from caramba.runtime.plan import RuntimePlan
from caramba.runtime.plan.builder import RuntimePlanBuilder
from caramba.runtime.tensordict_utils import (
    TensorDictBase,
    as_tensordict,
    to_device,
)
from caramba.trainer.mosaic_table2 import Table2SummaryWriter, Table2Telemetry
from caramba.trainer.objectives import MosaicNextTokenWithAuxObjective
from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from caramba.trainer.stepper import TrainSession, TrainStepper
from caramba.trainer.swap_manager import SwapManager
from caramba.trainer.train_dataloader.builder import TrainDataLoaderBuilder
from caramba.trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)


def _bytes_to_mb(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)


def _is_unexpected_ctx_kwarg_error(e: TypeError) -> bool:
    msg = str(e)
    return ("unexpected keyword argument" in msg) and ("ctx" in msg)


class _ForwardCaller:
    """Calls system.forward with an optional ctx, caching whether ctx is supported."""

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


class StandardTrainStepper(TrainStepper):
    def __init__(
        self,
        *,
        runtime_plan_builder: RuntimePlanBuilder | None = None,
        train_dataloader_builder: TrainDataLoaderBuilder | None = None,
    ) -> None:
        self._runtime_plan_builder = runtime_plan_builder or RuntimePlanBuilder()
        self._train_dataloader_builder = train_dataloader_builder or TrainDataLoaderBuilder()
        # Track loss from previous step for memory tuner metrics.
        self._last_step_loss: float | None = None
        # Track accuracy from previous step for memory tuner metrics.
        self._last_step_accuracy: float | None = None

    def run(self, session: TrainSession) -> None:
        defaults = session.defaults
        target = session.target
        run = session.run
        train = session.train
        dataset_comp = session.dataset_comp
        system = session.system
        objective = session.objective
        checkpoint_dir = session.checkpoint_dir
        run_logger = session.run_logger

        torch.manual_seed(run.seed)
        device = torch.device(train.device)

        # Reset shared tuner for new training runs.
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

        # W&B writer (optional).
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

        # CUDA perf knobs.
        if device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception as e:
                logger.fallback_warning(
                    "WARNING: Failed to enable TF32; continuing without TF32. "
                    f"device={device.type} dtype={dtype} error={e}"
                )

        # Wrap system module for DDP/FSDP if requested.
        if dist_ctx is not None:
            try:
                system_any = cast(Any, system)
                m = getattr(system_any, "module", None)
                if isinstance(m, nn.Module):
                    system_any.module = dist_ctx.wrap_model(m)
            except Exception as e:
                raise RuntimeError("Failed to wrap system module for DDP/FSDP") from e

        compiled = False

        loader = self._build_loader(
            dataset_comp=dataset_comp,
            defaults=defaults,
            train=train,
            device=device,
            batch_size=int(runtime_plan.batch_size),
            dist_ctx=dist_ctx,
        )

        # Reproducibility artifacts (export IO shapes BEFORE torch.compile).
        self._export_compiled_plan(checkpoint_dir=checkpoint_dir, target=target)
        self._export_io_shapes(
            checkpoint_dir=checkpoint_dir,
            loader=loader,
            system=system,
            device=device,
        )

        # Activation checkpointing.
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

        # Optional torch.compile.
        if bool(getattr(runtime_plan, "compile", False)):
            try:
                system_any = cast(Any, system)
                m = getattr(system_any, "module", None)
                if isinstance(m, nn.Module):
                    compile_mode = str(getattr(runtime_plan, "compile_mode", "default"))
                    if device.type == "cuda":
                        accum_steps_cfg = max(
                            1, int(getattr(train, "gradient_accumulation_steps", 1) or 1)
                        )
                        if compile_mode == "reduce-overhead" and int(accum_steps_cfg) > 1:
                            compile_mode = "default"
                            logger.info(
                                "torch.compile mode reduced from 'reduce-overhead' to 'default' because "
                                f"gradient_accumulation_steps={accum_steps_cfg} can trip CUDA-graph output reuse."
                            )
                        if compile_mode == "max-autotune" and int(accum_steps_cfg) > 1:
                            compile_mode = "max-autotune-no-cudagraphs"
                            logger.info(
                                "torch.compile mode adjusted from 'max-autotune' to 'max-autotune-no-cudagraphs' because "
                                f"gradient_accumulation_steps={accum_steps_cfg} requires avoiding CUDA-graph output reuse."
                            )
                    try:
                        system_any.module = torch.compile(m, mode=str(compile_mode))
                    except Exception as e:
                        if str(compile_mode) == "max-autotune-no-cudagraphs":
                            logger.warning(
                                "torch.compile mode 'max-autotune-no-cudagraphs' failed; falling back to 'default'. "
                                f"Error: {type(e).__name__}: {e}"
                            )
                            system_any.module = torch.compile(m, mode="default")
                        else:
                            raise
                    compiled = True
                    logger.info(f"torch.compile enabled (mode={compile_mode}) for {target.name}:{run.id}")
                else:
                    logger.info(
                        "torch.compile requested but no nn.Module found on system.module "
                        f"(target={target.name} run={run.id}); continuing without compile."
                    )
            except Exception as e:
                raise RuntimeError("Failed to compile model") from e

        optimizer = build_optimizer(
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
        debug_microbatches = bool(getattr(train, "debug_microbatches", False))
        debug_microbatch_sync = bool(getattr(train, "debug_microbatch_sync", False))

        def _maybe_sync_debug() -> None:
            if not debug_microbatch_sync:
                return
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

        telemetry_interval = int(getattr(train, "telemetry_interval", 10)) or 10
        telemetry_interval = max(1, telemetry_interval)

        profile_every = int(getattr(train, "profile_every", 0)) or 0
        profile_every = max(0, profile_every)
        profile_record_shapes = bool(getattr(train, "profile_record_shapes", False))

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
            raise RuntimeError("Failed to write run telemetry event") from e

        if not hasattr(system, "forward"):
            raise TypeError("System component does not expose forward()")
        forward_caller = _ForwardCaller(system.forward)  # type: ignore[attr-defined]

        # Objective loss/metrics signatures (support both (batch) and (batch_td) styles).
        if not hasattr(objective, "loss"):
            raise TypeError("Objective component does not expose loss()")
        loss_fn = objective.loss  # type: ignore[attr-defined]
        batch_kw = self._resolve_loss_batch_key(loss_fn)

        def call_objective_loss(*, outputs: object, batch_td: TensorDictBase) -> Tensor:
            kwargs: dict[str, object] = {"outputs": outputs}
            kwargs[batch_kw] = batch_td
            return loss_fn(**kwargs)  # type: ignore[misc]

        call_objective_metrics = self._make_objective_metrics_caller(objective)

        # Training-time hooks (composable concerns).
        table2 = Table2Telemetry()
        table2_writer = Table2SummaryWriter()
        hooks = list(session.hooks or [])

        hooks.append(
            RunLoggerHook(run_logger=run_logger, run_id=str(run.id), phase="standard")
        )
        if wandb_writer is not None:
            hooks.append(WandBHook(writer=wandb_writer))

        hooks.append(VizHook(run_logger=run_logger, run_id=str(run.id), phase="standard"))

        layer_interval = int(getattr(train, "layer_telemetry_interval", 0) or 0)
        if layer_interval > 0:
            hooks.append(
                LayerStatsHook(
                    system=system,
                    interval=int(layer_interval),
                    run_logger=run_logger,
                    run_id=str(run.id),
                    phase="standard",
                )
            )

        hooks.append(
            Table2ExportHook(
                enabled=True,
                checkpoint_dir=checkpoint_dir,
                target=target,
                run=run,
                system=system,
                dataset_comp=dataset_comp,
                table2_writer=table2_writer,
                table2_cfg=table2.cfg,
                run_logger=run_logger,
            )
        )
        hooks.append(
            FinalCheckpointHook(
                checkpoint_dir=checkpoint_dir,
                run_id=str(run.id),
                phase="standard",
                system=system,
            )
        )

        loader_iter = iter(loader)
        for h in hooks:
            h.on_run_begin()

        try:
            with logger.progress_bar() as progress:
                task = progress.add_task("Training...", total=int(run.steps))
                step_task = progress.add_task("Waiting...", total=None)

                for step0 in range(int(run.steps)):
                    step_1 = int(step0 + 1)
                    t_step0 = time.perf_counter()
                    data_time_s = 0.0
                    fwd_bwd_time_s = 0.0

                    # Viz context setup.
                    viz_interval = max(0, int(getattr(train, "viz_interval", 0) or 0))
                    viz_enabled = bool(viz_interval > 0 and (step_1 % viz_interval) == 0)

                    viz_ctx = TrainingVizMosaicContext(
                        enabled=bool(viz_enabled),
                        step=int(step_1),
                        max_tokens=int(getattr(train, "viz_tokens", 16) or 16),
                        max_channels=int(getattr(train, "viz_channels", 32) or 32),
                        max_heads=int(getattr(train, "viz_heads", 4) or 4),
                        topk=int(getattr(train, "viz_topk", 8) or 8),
                    )

                    viz_ctx._last_loss = self._last_step_loss
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

                    accum_steps = max(1, int(getattr(train, "gradient_accumulation_steps", 1) or 1))

                    progress.update(
                        step_task,
                        total=int(accum_steps),
                        completed=0,
                        description=f"Step {step_1} • Preparing...",
                        visible=True,
                    )

                    for h in hooks:
                        h.on_step_begin(step=int(step_1))

                    optimizer.zero_grad(set_to_none=True)

                    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                    outputs_last: object | None = None
                    loss_last: Tensor | None = None
                    last_batch_td: TensorDictBase | None = None
                    kernel_events_estimate: int | None = None

                    did_profile_first = bool(profile_every > 0 and (step_1 % profile_every) == 0)
                    did_profiled = False
                    prof = None
                    for mb_i in range(int(accum_steps)):
                        t_data_mb0 = time.perf_counter()
                        progress.update(
                            step_task,
                            description=f"Step {step_1} • Microbatch {mb_i+1}/{accum_steps} (Fetching...)",
                        )
                        item, loader_iter = self._next_loader_item(loader, loader_iter)
                        if isinstance(item, TensorDictBase):
                            batch_td = item
                        elif isinstance(item, dict):
                            batch_td = as_tensordict(item)
                        else:
                            raise TypeError(
                                "Trainer expects batch items to be dict/TensorDict. "
                                f"Got {type(item).__name__}."
                            )
                        t_h2d0 = time.perf_counter() if debug_microbatches else 0.0
                        mb = cast(
                            TensorDictBase,
                            to_device(
                                batch_td,
                                device=device,
                                non_blocking=bool(
                                    getattr(train, "pin_memory", False) and device.type == "cuda"
                                ),
                            ),
                        )
                        if debug_microbatches:
                            _maybe_sync_debug()
                            dt_ms = (time.perf_counter() - t_h2d0) * 1e3
                            logger.info(
                                f"[DEBUG] Step {step_1} MB {mb_i+1}: Fetching+H2D DONE ({dt_ms:.1f} ms)"
                            )
                        data_time_s += float(time.perf_counter() - t_data_mb0)

                        if compiled and device.type == "cuda":
                            try:
                                torch.compiler.cudagraph_mark_step_begin()
                            except Exception as e:
                                raise RuntimeError(
                                    "torch.compile is enabled on CUDA, but failed to mark cudagraph step boundary."
                                ) from e

                        t_fwb_mb0 = time.perf_counter()
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
                                    )
                                    loss_sum += loss_last.detach().float()
                                    (loss_last / float(accum_steps)).backward()
                                prof = prof_ctx
                                did_profiled = True
                            except Exception as e:
                                raise RuntimeError("Failed to run torch profiler") from e
                        else:
                            outputs_last, loss_last = self._forward_loss(
                                batch_td=mb,
                                device=device,
                                use_amp=use_amp,
                                amp_dtype=amp_dtype,
                                viz_ctx=viz_ctx,
                                forward_caller=forward_caller,
                                objective_loss=call_objective_loss,
                            )
                            if not isinstance(loss_last, Tensor):
                                raise TypeError(
                                    f"Objective.loss must return a Tensor, got {type(loss_last).__name__}"
                                )
                            loss_sum += loss_last.detach().float()
                            progress.update(
                                step_task,
                                description=f"Step {step_1} • Microbatch {mb_i+1}/{accum_steps} (Backward...)",
                            )
                            if debug_microbatches:
                                _maybe_sync_debug()
                                t_bwd0 = time.perf_counter()
                            (loss_last / float(accum_steps)).backward()
                            if debug_microbatches:
                                _maybe_sync_debug()
                                dt_ms = (time.perf_counter() - t_bwd0) * 1e3
                                logger.info(
                                    f"[DEBUG] Step {step_1} MB {mb_i+1}: Backward DONE ({dt_ms:.1f} ms)"
                                )

                        fwd_bwd_time_s += float(time.perf_counter() - t_fwb_mb0)
                        last_batch_td = mb
                        progress.update(step_task, advance=1)

                    if prof is not None:
                        try:
                            kernel_events_estimate = int(len(prof.events() or []))
                        except Exception as e:
                            raise RuntimeError("Failed to count profiler events") from e
                        try:
                            prof_dir = checkpoint_dir / "profiles"
                            prof_dir.mkdir(parents=True, exist_ok=True)
                            prof_path = prof_dir / f"profile_step_{step_1}.txt"

                            sort_key = "cpu_time_total"
                            if device.type == "cuda":
                                sort_key = "cuda_time_total"
                            elif device.type == "mps":
                                sort_key = "self_cpu_time_total"

                            table = prof.key_averages().table(
                                sort_by=sort_key,
                                row_limit=40,
                            )
                            out = [str(table)]
                            if profile_record_shapes:
                                out.append(
                                    prof.key_averages(group_by_input_shape=True).table(
                                        sort_by=sort_key,
                                        row_limit=80,
                                    )
                                )
                            prof_path.write_text("\n\n".join(out) + "\n", encoding="utf-8")
                            logger.info(f"Saved torch profiler summary to {prof_path}")
                        except Exception as e:
                            raise RuntimeError("Failed to export torch profiler summary") from e

                    progress.update(step_task, description=f"Step {step_1} • Optimizing...")
                    t_fwb1 = time.perf_counter()

                    clip = float(getattr(train, "grad_clip_norm", 0.0) or 0.0)
                    if clip > 0.0:
                        try:
                            torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                                system.parameters(),  # type: ignore[arg-type]
                                max_norm=float(clip),
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to clip gradients") from e

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
                    if debug_microbatches:
                        _maybe_sync_debug()
                        logger.info(f"[DEBUG] Step {step_1}: Optimizer DONE")

                    t_after_optim = time.perf_counter()

                    # Build step metrics only at the requested telemetry cadence.
                    metrics: dict[str, float] = {}
                    if (step_1 % telemetry_interval) == 0:
                        metrics = build_standard_step_metrics(
                            system=system,
                            train=train,
                            runtime_plan=runtime_plan,
                            optimizer=optimizer,
                            compiled=compiled,
                            step_time_s=float(max(1e-9, t_after_optim - t_step0)),
                            data_time_s=float(max(0.0, data_time_s)),
                            fwd_bwd_time_s=float(max(0.0, fwd_bwd_time_s)),
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

                        # Update persistent accuracy (loss is updated after forward pass).
                        acc_vals = [
                            v
                            for k, v in metrics.items()
                            if str(k).startswith("acc/bin_") and float(v) >= 0.0
                        ]
                        if acc_vals:
                            self._last_step_accuracy = float(sum(acc_vals) / len(acc_vals))
                            update_training_metrics(step=step_1, accuracy=self._last_step_accuracy)

                    extras: dict[str, Any] = {}
                    if viz_ctx.enabled:
                        extras["viz"] = viz_ctx.to_event()

                    for h in hooks:
                        h.on_step_end(
                            step=int(step_1),
                            metrics=metrics,
                            outputs=outputs_last,
                            batch=last_batch_td,
                            extras=extras or None,
                        )

                    desc = f"Step {step_1}/{run.steps}"
                    if metrics:
                        loss_val_live = float(metrics.get("loss", metrics.get("train_loss", 0.0)))
                        desc = f"{desc} • loss={float(loss_val_live):.4f}"
                    progress.update(task, advance=1, description=desc)

            for h in hooks:
                h.on_run_end(step=int(run.steps))

        finally:
            # Ensure hooks are cleaned up even on errors.
            for h in hooks:
                try:
                    h.close()
                except Exception as e:
                    raise RuntimeError("Failed to cleanup training hooks") from e

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
    ) -> tuple[object, Tensor]:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            inp = batch_td.get("input_ids", None)  # type: ignore[attr-defined]
            if isinstance(inp, Tensor):
                viz_ctx.input_ids = inp

            dl = batch_td.get("mosaic_drop_local", None)  # type: ignore[attr-defined]
            if isinstance(dl, Tensor):
                viz_ctx.mosaic_drop_local = dl

            teacher: dict[str, Tensor] = {}
            for k in ("read_bucket", "write_bucket", "write_gate", "clear"):
                v = batch_td.get(f"memblock_teacher_{k}", None)  # type: ignore[attr-defined]
                if isinstance(v, Tensor):
                    teacher[k] = v
            viz_ctx.memblock_teacher = teacher or None

            collect_aux = bool(teacher)
            try:
                if isinstance(getattr(objective_loss, "__self__", None), MosaicNextTokenWithAuxObjective):
                    collect_aux = True
            except Exception as e:
                raise RuntimeError("Failed to check if objective is MosaicNextTokenWithAuxObjective") from e

            viz_ctx.memblock_collect_aux = bool(collect_aux)

            ctx_to_pass: object | None = viz_ctx if (bool(viz_ctx.enabled) or bool(collect_aux)) else None
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
            if not isinstance(loss, Tensor):
                raise TypeError(f"Objective.loss must return a Tensor, got {type(loss).__name__}")

            self._last_step_loss = float(loss.detach())
            update_training_metrics(step=viz_ctx.step, loss=self._last_step_loss)
            return outputs, loss

    @staticmethod
    def _compute_memblock_teacher_p(*, train: TrainConfig, step_1: int, total_steps: int) -> float:
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

    def _next_loader_item(self, loader: DataLoader, it: Iterable[Any]) -> tuple[Any, Iterable[Any]]:
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

    @staticmethod
    def _resolve_loss_batch_key(loss_fn: Any) -> str:
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

    @staticmethod
    def _make_objective_metrics_caller(objective: object) -> Any:
        if not hasattr(objective, "metrics"):
            def _none(*_a: object, **_k: object) -> dict[str, float] | None:
                return None
            return _none

        metrics_fn = objective.metrics  # type: ignore[attr-defined]
        try:
            metrics_sig = inspect.signature(metrics_fn)
        except Exception as e:
            raise RuntimeError("Failed to introspect objective metrics function") from e

        metrics_params = metrics_sig.parameters
        metrics_accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in metrics_params.values()
        )

        def _call(*, outputs: object, batch_td: TensorDictBase, loss: Tensor) -> dict[str, float] | None:
            kwargs: dict[str, object] = {}

            if "outputs" in metrics_params:
                kwargs["outputs"] = outputs
            elif "_outputs" in metrics_params:
                kwargs["_outputs"] = outputs
            elif metrics_accepts_kwargs:
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

        return _call

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

            b_export = b0[:1]

            if not hasattr(system, "forward"):
                raise TypeError(
                    "Failed to export IO shapes: system has no forward().\n"
                    "Fix: ensure the system object exposes forward(batch, ...) so we can capture output shapes."
                )

            fwd = system.forward  # type: ignore[attr-defined]
            sig = inspect.signature(fwd)
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

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        dt = str(dtype).lower()
        if dt == "float32":
            return torch.float32
        if dt == "float16":
            return torch.float16
        if dt == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _enable_activation_checkpointing(self_mod: nn.Module, *, enabled: bool, threshold_mb: float) -> None:
        for m in self_mod.modules():
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

    @staticmethod
    def _param_bytes(system: object) -> int:
        total = 0
        for p in system.parameters():  # type: ignore[attr-defined]
            if isinstance(p, Tensor):
                total += int(p.numel()) * int(p.element_size())
        return total
