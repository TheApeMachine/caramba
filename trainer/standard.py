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
from pathlib import Path
import time
import math
from typing import Any, Protocol, Sized, cast

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from caramba.carmath import (
    autocast_dtype,
    autocast_dtype_str,
    safe_perplexity_from_nll,
    token_budget_batch_size,
    train_val_counts,
    weight_dtype_str,
)
from caramba.config.defaults import Defaults
from caramba.config.manifest import Manifest
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.console import logger
from caramba.instrumentation import RunLogger, TrainingVizContext
from caramba.instrumentation.viz import TrainingVizMosaicContext
from caramba.instrumentation.wandb_writer import WandBWriter
from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from caramba.runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan
from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict, collate_tensordict, to_device
from caramba.layer.attention import AttentionLayer
from caramba.layer.mosaic.block import MosaicBlockLayer
from caramba.trainer.swap_manager import SwapManager


class _Engine(Protocol):
    registry: Any


class StandardTrainer:
    def __init__(self, *, checkpoint_dir: str | None = None) -> None:
        self._checkpoint_dir_override = checkpoint_dir

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

        # Build components once per target.
        dataset_comp = engine.registry.build(target.data, backend=str(target.backend))
        system = engine.registry.build(target.system, backend=str(target.backend))
        objective = engine.registry.build(target.objective, backend=str(target.backend))

        ckpt_dir = (
            Path(self._checkpoint_dir_override)
            if self._checkpoint_dir_override
            else Path("runs") / target.name
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Note: when running distributed, only rank0 should write logs.
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
        torch.manual_seed(run.seed)
        device = torch.device(train.device)

        dist_ctx = None
        dist_strategy = str(getattr(train, "distributed_strategy", "none")).lower()
        if dist_strategy != "none":
            try:
                from caramba.trainer.distributed import DistributedConfig, DistributedContext, DistributedStrategy

                cfg = DistributedConfig(
                    strategy=DistributedStrategy(dist_strategy),
                    backend=str(getattr(train, "distributed_backend", "nccl")),
                )
                dist_ctx = DistributedContext.init(cfg)
                device = dist_ctx.device
            except Exception as e:
                raise RuntimeError(f"Failed to initialize distributed training: {e}") from e
            # Only rank0 writes JSONL/HDF5 logs.
            try:
                if hasattr(dist_ctx, "is_main") and not bool(getattr(dist_ctx, "is_main")):
                    run_logger.enabled = False
            except Exception as e:
                raise RuntimeError("Failed to check if this is the main process") from e

        # Optional W&B writer (standard trainer previously only emitted JSONL).
        # This is best-effort and will never crash training.
        wandb_writer: WandBWriter | None = None
        if bool(getattr(defaults.logging, "wandb", False)):
            is_main = True
            if dist_ctx is not None:
                try:
                    if hasattr(dist_ctx, "is_main"):
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
                    raise RuntimeError("Failed to initialize W&B writer") from e

        runtime_plan = self._load_or_create_runtime_plan(
            checkpoint_dir=checkpoint_dir,
            device=device,
            train=train,
            system_cfg=dict(target.system.config),
        )
        dtype = self._parse_dtype(runtime_plan.dtype)

        if hasattr(system, "to"):
            system.to(device=device, dtype=dtype)  # type: ignore[attr-defined]

        # Wrap system module for DDP/FSDP if requested.
        if dist_ctx is not None:
            try:
                m = getattr(system, "module", None)
                if isinstance(m, torch.nn.Module):
                    system.module = dist_ctx.wrap_model(m)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError("Failed to wrap system module for DDP/FSDP") from e

        compiled = False
        if bool(getattr(runtime_plan, "compile", False)):
            # Best-effort torch.compile: only apply when a module is exposed.
            try:
                m = getattr(system, "module", None)
                if isinstance(m, torch.nn.Module):
                    system.module = torch.compile(m, mode=str(runtime_plan.compile_mode))  # type: ignore[attr-defined]
                    compiled = True
            except Exception as e:
                raise RuntimeError("Failed to compile model") from e

        loader = self._build_loader(
            dataset_comp=dataset_comp,
            defaults=defaults,
            train=train,
            device=device,
            batch_size=int(runtime_plan.batch_size),
            dist_ctx=dist_ctx,
        )

        if not hasattr(system, "parameters"):
            raise TypeError("System component does not expose parameters()")
        opt_name = str(getattr(train, "optimizer", "adamw")).lower()
        weight_decay = float(getattr(train, "weight_decay", 0.0))
        fused_opt = bool(getattr(train, "fused_optimizer", False))
        if opt_name in ("adamw", "adam"):
            optimizer = torch.optim.AdamW(
                system.parameters(),  # type: ignore[arg-type]
                lr=float(train.lr),
                weight_decay=float(weight_decay),
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                system.parameters(),  # type: ignore[arg-type]
                lr=float(train.lr),
                weight_decay=float(weight_decay),
            )
        elif opt_name == "lion":
            from caramba.optimizer.lion import Lion

            optimizer = Lion(
                system.parameters(),  # type: ignore[arg-type]
                lr=float(train.lr),
                weight_decay=float(weight_decay),
                fused=bool(fused_opt),
            )
        else:
            raise ValueError(f"Unknown optimizer {opt_name!r}")

        # Optional LR scheduler (manifest-driven).
        lr_scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        from caramba.trainer.swap_manager import SwapManager

        swap = SwapManager(
            offload_optimizer=bool(getattr(train, "offload_optimizer", False)),
            offload_device="cpu",
        )

        use_amp = bool(train.use_amp)
        amp_dtype = autocast_dtype(device, str(train.amp_dtype))

        telemetry_interval = int(getattr(train, "telemetry_interval", 10)) or 10
        profile_every = int(getattr(train, "profile_every", 0)) or 0
        profile_record_shapes = bool(getattr(train, "profile_record_shapes", False))
        layer_stats_enabled = False
        layer_telemetry_interval = int(telemetry_interval)
        _hook_handles: list[RemovableHandle] = []
        attn_modules: list[tuple[int, str, torch.nn.Module]] = []
        mosaic_modules: list[tuple[int, str, torch.nn.Module]] = []

        class _LayerStatsCollector:
            enabled: bool

            def __init__(self) -> None:
                self.enabled = False
                self.reset()

            def reset(self) -> None:
                self.counts: dict[int, int] = {}
                self.sum_ms: dict[int, float] = {}
                self.sum_ma: dict[int, float] = {}
                self.max_abs: dict[int, float] = {}
                self.shapes: dict[int, list[int]] = {}

            def observe(self, idx: int, y: Tensor) -> None:
                # Keep overhead bounded; this is best-effort.
                try:
                    z = y.detach()
                    z = z.float()
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
                    raise RuntimeError("Failed to observe layer stats") from e

            def finalize(self) -> list[dict[str, object]]:
                out: list[dict[str, object]] = []
                for idx, name, mod in attn_modules:
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
                                getattr(
                                    getattr(mod, "mode", None),
                                    "value",
                                    getattr(mod, "mode", ""),
                                )
                            ),
                            "null_attn": bool(getattr(cfg, "null_attn", False)) if cfg is not None else False,
                            "tie_qk": bool(getattr(cfg, "tie_qk", False)) if cfg is not None else False,
                            "rope_semantic": bool(getattr(cfg, "rope_semantic", False)) if cfg is not None else False,
                            "decoupled_gate": bool(getattr(cfg, "decoupled_gate", False)) if cfg is not None else False,
                        }
                    )
                return out

        _collector = _LayerStatsCollector()

        try:
            raw = getattr(train, "layer_telemetry_interval", None)
            # Default to every step (metrics telemetry can remain slower).
            layer_telemetry_interval = int(raw) if raw is not None else 1
            layer_telemetry_interval = max(1, int(layer_telemetry_interval))

            root_mod = getattr(system, "module", None)
            if isinstance(root_mod, nn.Module):
                try:
                    modules_iter = root_mod.named_modules()
                except Exception as e:
                    raise RuntimeError(f"Failed to iterate over model modules: {e}") from e

                for name, mod in modules_iter:
                    if isinstance(mod, AttentionLayer):
                        # Give the module a stable viz id/name so it can emit per-layer samples.
                        try:
                            setattr(mod, "_viz_index", int(len(attn_modules)))
                            setattr(mod, "_viz_name", str(name))
                        except Exception as e:
                            raise RuntimeError(f"Failed to set attributes on attention layer {name!r}: {e}") from e

                        attn_modules.append((len(attn_modules), str(name), mod))
                    if isinstance(mod, MosaicBlockLayer):
                        try:
                            setattr(mod, "_mosaic_index", int(len(mosaic_modules)))
                            setattr(mod, "_mosaic_name", str(name))
                        except Exception as e:
                            raise RuntimeError(f"Failed to set attributes on mosaic layer {name!r}: {e}") from e

                        mosaic_modules.append((len(mosaic_modules), str(name), mod))

                def _make_hook(i: int):
                    def _hook(_m: nn.Module, _inp: tuple[object, ...], out: object) -> None:
                        if not _collector.enabled:
                            return
                        y = (
                            out[0]
                            if isinstance(out, tuple)
                            and len(out) > 0
                            and isinstance(out[0], Tensor)
                            else out
                        )
                        if isinstance(y, Tensor):
                            _collector.observe(i, y)

                    return _hook

                for idx, _name, mod in attn_modules:
                    try:
                        handle = mod.register_forward_hook(_make_hook(idx))
                        _hook_handles.append(handle)
                    except Exception as e:
                        raise RuntimeError(f"Failed to register forward hook on attention layer {_name!r} (index {idx}): {e}") from e

                layer_stats_enabled = len(attn_modules) > 0
        except RuntimeError:
            # Re-raise RuntimeError as-is (already has context)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to register forward hooks: {e}") from e

        def bytes_to_mb(n: int) -> float:
            return float(n) / (1024.0 * 1024.0)

        def param_bytes() -> int:
            total = 0
            for p in system.parameters():  # type: ignore[attr-defined]
                if isinstance(p, Tensor):
                    total += int(p.numel()) * int(p.element_size())
            return total

        def grad_bytes() -> int:
            total = 0
            for p in system.parameters():  # type: ignore[attr-defined]
                g = getattr(p, "grad", None)
                if isinstance(g, Tensor):
                    total += int(g.numel()) * int(g.element_size())
            return total

        def optim_state_bytes() -> int:
            total = 0
            try:
                for _param, state in optimizer.state.items():
                    if isinstance(state, dict):
                        for v in state.values():
                            if isinstance(v, Tensor):
                                total += int(v.numel()) * int(v.element_size())
            except Exception as e:
                raise RuntimeError("Failed to compute optimizer state bytes") from e

            return total

        # Emit static-ish run metadata once (best-effort).
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
                    "param_mb": bytes_to_mb(param_bytes()),
                },
            )
        except Exception as e:
            raise RuntimeError("Failed to emit telemetry") from e

        # Export a reproducibility artifact (lowered plan + io shapes).
        try:
            from caramba.compiler.plan import Planner

            plan_txt = Planner().format_target(target, indent=0, path=f"targets[{target.name}]")
            (checkpoint_dir / "compiled_plan.txt").write_text("\n".join(plan_txt) + "\n", encoding="utf-8")
        except Exception:
            logger.warning("Failed to export compiled plan (ignoring)")
        try:
            # Capture one batch IO signature (best-effort).
            it = iter(loader)
            b0 = next(it)
            b0 = b0 if isinstance(b0, TensorDictBase) else as_tensordict(b0)  # type: ignore[arg-type]
            b0 = cast(TensorDictBase, to_device(b0, device=device))
            if hasattr(system, "forward"):
                o0 = system.forward(b0)  # type: ignore[attr-defined]
            else:
                o0 = {}
            def shape_sig(td: object) -> dict[str, object]:
                out: dict[str, object] = {}
                if isinstance(td, dict):
                    items = td.items()
                else:
                    try:
                        items = dict(td).items()  # type: ignore[arg-type]
                    except Exception as e:
                        raise RuntimeError("Failed to introspect IO shapes") from e

                for k, v in items:
                    if isinstance(v, Tensor):
                        out[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                return out
            import json
            (checkpoint_dir / "io_shapes.json").write_text(
                json.dumps({"batch": shape_sig(b0), "outputs": shape_sig(o0)}, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            raise RuntimeError("Failed to export IO shapes") from e

        logger.header("Training", f"{target.name}:{run.id} • {run.steps} steps")
        loader_iter = iter(loader)

        def _call_objective_loss(*, outputs: object, batch_td: TensorDictBase) -> Tensor:
            if not hasattr(objective, "loss"):
                raise TypeError("Objective component does not expose loss()")
            loss_fn = objective.loss  # type: ignore[attr-defined]
            try:
                sig = inspect.signature(loss_fn)
                params = sig.parameters
            except Exception as e:
                raise RuntimeError("Failed to introspect objective loss function") from e

            kwargs: dict[str, object] = {}

            if "outputs" in params:
                kwargs["outputs"] = outputs
            else:
                # All current objectives use `outputs=...`; keep this strict.
                raise TypeError("Objective.loss must accept keyword argument 'outputs'")

            if "batch" in params:
                kwargs["batch"] = batch_td
            elif "_batch" in params:
                kwargs["_batch"] = batch_td
            elif "batch_td" in params:
                kwargs["batch_td"] = batch_td
            elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                # Best-effort: prefer the canonical name if **kwargs is accepted.
                kwargs["batch"] = batch_td
            else:
                raise TypeError("Objective.loss must accept a batch keyword (e.g. 'batch' or '_batch')")

            loss = loss_fn(**kwargs)

            if not isinstance(loss, Tensor):
                raise TypeError(f"Objective.loss must return a Tensor, got {type(loss).__name__}")

            return loss

        def _call_objective_metrics(
            *, outputs: object, batch_td: TensorDictBase, loss: Tensor
        ) -> dict[str, float] | None:
            if not hasattr(objective, "metrics"):
                return None

            metrics_fn = objective.metrics  # type: ignore[attr-defined]

            try:
                sig = inspect.signature(metrics_fn)
                params = sig.parameters
            except Exception as e:
                raise RuntimeError("Failed to introspect objective metrics function") from e

            kwargs: dict[str, object] = {}
            if "outputs" in params:
                kwargs["outputs"] = outputs
            if "batch" in params:
                kwargs["batch"] = batch_td
            elif "_batch" in params:
                kwargs["_batch"] = batch_td
            elif "batch_td" in params:
                kwargs["batch_td"] = batch_td
            elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                kwargs["batch"] = batch_td

            if "loss" in params:
                kwargs["loss"] = loss
            elif "_loss" in params:
                kwargs["_loss"] = loss
            elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                kwargs["loss"] = loss

            extra = metrics_fn(**kwargs)
            return cast(dict[str, float] | None, extra) if isinstance(extra, dict) else None

        def _forward_loss(batch_td: TensorDictBase) -> tuple[object, Tensor]:
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                if not hasattr(system, "forward"):
                    raise TypeError("System component does not expose forward()")
                try:
                    # Attach MOSAIC-friendly fields onto the ctx (best-effort).
                    # Note: TrainingVizContext uses slots, so we use a subclass that adds fields.
                    if isinstance(viz_ctx, TrainingVizMosaicContext):
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
                                v = batch_td.get(f"mosaic_teacher_{k}", None)  # type: ignore[attr-defined]
                            except Exception as e:
                                raise RuntimeError("Failed to get mosaic teacher signal") from e

                            if isinstance(v, Tensor):
                                teacher[k] = v

                        viz_ctx.mosaic_teacher = teacher or None

                        collect_aux = bool(teacher)
                        try:
                            from caramba.trainer.objectives import MosaicNextTokenWithAuxObjective

                            if isinstance(objective, MosaicNextTokenWithAuxObjective):
                                collect_aux = True
                        except Exception as e:
                            raise RuntimeError("Failed to check if objective expects MOSAIC aux") from e

                        viz_ctx.mosaic_collect_aux = bool(collect_aux)
                    outputs = system.forward(batch_td, ctx=viz_ctx)  # type: ignore[attr-defined]
                except TypeError:
                    # Some non-standard systems may not accept ctx; keep best-effort.
                    outputs = system.forward(batch_td)  # type: ignore[attr-defined]
                # Best-effort: merge MOSAIC aux outputs from ctx into outputs so
                # objectives can see them even when the system doesn't attach them.
                try:
                    aux_out = getattr(viz_ctx, "mosaic_aux_out", None)
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

                loss = _call_objective_loss(outputs=outputs, batch_td=batch_td)
            return outputs, loss

        try:
            with logger.progress_bar() as progress:
                task = progress.add_task("Training...", total=int(run.steps))
                for step in range(int(run.steps)):
                    t0 = time.perf_counter()
                    # Training viz: emit small attention/activation samples periodically.
                    # Default to every step; this is small (downsampled) and users
                    # often want responsive visuals even when steps are slow.
                    viz_interval = int(getattr(train, "viz_interval", 1) or 1)
                    viz_interval = max(1, int(viz_interval))
                    viz_enabled = ((step + 1) % viz_interval) == 0
                    viz_ctx = TrainingVizMosaicContext(
                        enabled=bool(viz_enabled),
                        step=int(step + 1),
                        max_tokens=int(getattr(train, "viz_tokens", 16) or 16),
                        max_channels=int(getattr(train, "viz_channels", 32) or 32),
                        max_heads=int(getattr(train, "viz_heads", 4) or 4),
                        topk=int(getattr(train, "viz_topk", 8) or 8),
                    )
                    # MOSAIC scheduled sampling: compute teacher mixing probability.
                    try:
                        total = int(run.steps)
                        s = int(step + 1)
                        warm = int(getattr(train, "mosaic_teacher_p_warmup_steps", 0) or 0)
                        cool = int(getattr(train, "mosaic_teacher_p_cooldown_steps", 0) or 0)
                        p0 = float(getattr(train, "mosaic_teacher_p_start", 1.0))
                        p1 = float(getattr(train, "mosaic_teacher_p_end", 0.0))
                        sched = str(getattr(train, "mosaic_teacher_p_schedule", "linear")).lower()
                        # Clamp phases.
                        s_eff = max(0, min(total, s))
                        # Linear interpolation over the "middle" region.
                        denom = max(1, total - warm - cool)
                        if s_eff <= warm:
                            alpha = 0.0
                        elif s_eff >= total - cool:
                            alpha = 1.0
                        else:
                            alpha = float(s_eff - warm) / float(denom)
                        if sched == "cosine":
                            import math as _math

                            alpha2 = 0.5 - 0.5 * _math.cos(_math.pi * float(alpha))
                        elif sched == "constant":
                            alpha2 = 0.0
                        else:
                            alpha2 = float(alpha)
                        p_t = float(p0 + (p1 - p0) * float(alpha2))
                        viz_ctx.mosaic_teacher_p = float(max(0.0, min(1.0, p_t)))
                    except Exception as e:
                        raise RuntimeError("Failed to compute mosaic teacher p") from e

                    # MOSAIC scalar stats: compute only on metric logging steps.
                    try:
                        viz_ctx.mosaic_stats_enabled = bool(((step + 1) % int(telemetry_interval)) == 0)
                    except Exception as e:
                        raise RuntimeError("Failed to compute mosaic stats enabled") from e

                    accum_steps = int(getattr(train, "gradient_accumulation_steps", 1))
                    accum_steps = max(1, accum_steps)
                    optimizer.zero_grad(set_to_none=True)
                    kernel_launches: int | None = None
                    loss_sum = 0.0
                    outputs: object | None = None
                    last_batch_td: TensorDictBase | None = None

                    # Fetch/prepare microbatches for this optimizer step.
                    micro_batches: list[TensorDictBase] = []
                    t_data0 = time.perf_counter()
                    for _micro in range(int(accum_steps)):
                        try:
                            item = next(loader_iter)
                        except StopIteration:
                            raise RuntimeError("Reached end of loader, resetting")

                        if isinstance(item, TensorDictBase):
                            batch_td = item
                        elif isinstance(item, dict):
                            batch_td = as_tensordict(item)
                        else:
                            raise TypeError(
                                "StandardTrainer expects batch items to be dict/TensorDict. "
                                f"Got {type(item).__name__}."
                            )
                        batch_td = cast(TensorDictBase, to_device(batch_td, device=device))
                        micro_batches.append(batch_td)
                    t_data = time.perf_counter()

                    if layer_stats_enabled:
                        _collector.reset()
                        _collector.enabled = bool(((step + 1) % layer_telemetry_interval) == 0)

                    # Optional profiling (best-effort; primarily useful on CUDA).
                    if profile_every > 0 and ((step + 1) % profile_every == 0):
                        try:
                            from torch.profiler import ProfilerActivity, profile

                            acts = [ProfilerActivity.CPU]
                            if device.type == "cuda":
                                acts.append(ProfilerActivity.CUDA)
                            with profile(
                                activities=acts,
                                record_shapes=bool(profile_record_shapes),
                                profile_memory=True,
                            ) as prof:
                                # Profile only the first microbatch to keep overhead bounded.
                                outputs, loss = _forward_loss(micro_batches[0])
                                loss_sum += float(loss.detach())
                                (loss / float(accum_steps)).backward()
                            # Heuristic: count device-side events as “launch-ish”.
                            if device.type == "cuda":
                                try:
                                    evs = prof.events() or []
                                    # We intentionally keep this a heuristic; different builds expose
                                    # different event metadata. The main goal is “fewer events over time”.
                                    kernel_launches = int(len(evs))
                                except Exception as e:
                                    raise RuntimeError("Failed to count kernel launches") from e
                        except Exception as e:
                            raise RuntimeError("Failed to profile") from e

                    t_fwd = time.perf_counter()
                    # Backward over remaining microbatches (and the first when not profiled).
                    did_profile_first = bool(profile_every > 0 and ((step + 1) % profile_every == 0))
                    start_idx = 1 if did_profile_first else 0
                    for mb in micro_batches[start_idx:]:
                        outputs, loss = _forward_loss(mb)
                        loss_sum += float(loss.detach())
                        (loss / float(accum_steps)).backward()
                        last_batch_td = mb
                    if layer_stats_enabled:
                        _collector.enabled = False
                    if last_batch_td is None:
                        last_batch_td = micro_batches[-1]
                    t_bwd = time.perf_counter()
                    swap.before_optimizer_step(optimizer, device=device)
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    swap.after_optimizer_step(optimizer)
                    if bool(getattr(train, "offload_optimizer", False)):
                        # After offloading, grads should be freed aggressively.
                        optimizer.zero_grad(set_to_none=True)
                    t_optim = time.perf_counter()

                    # Emit layer-level telemetry independent of `telemetry_interval`.
                    if layer_stats_enabled and ((step + 1) % layer_telemetry_interval == 0):
                        try:
                            run_logger.log_event(
                                type="layer_stats",
                                run_id=str(run.id),
                                phase="standard",
                                step=step + 1,
                                data={"layers": _collector.finalize()},
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to log layer stats") from e

                    # Emit viz payload independent of `telemetry_interval`.
                    if viz_ctx.enabled:
                        try:
                            run_logger.log_event(
                                type="viz",
                                run_id=str(run.id),
                                phase="standard",
                                step=step + 1,
                                data=viz_ctx.to_event(),
                            )
                        except Exception as e:
                            raise RuntimeError("Failed to log layer stats") from e

                    if (step + 1) % telemetry_interval == 0:
                        # Log averaged loss per optimizer step (matches grad accumulation semantics).
                        loss_val = float(loss_sum) / float(accum_steps)
                        # Perplexity guardrails (avoid overflow in exp()).
                        ppl = float(safe_perplexity_from_nll(float(loss_val)))

                        lr = float(optimizer.param_groups[0].get("lr", float(train.lr)))
                        lr_base = float(getattr(train, "lr", lr))
                        lr_mult = (lr / lr_base) if lr_base > 0 else 1.0
                        grad_norm = 0.0
                        try:
                            from caramba.carmath import global_grad_norm_l2

                            grad_norm = float(global_grad_norm_l2(system))  # type: ignore[arg-type]
                        except Exception as e:
                            raise RuntimeError("Failed to compute gradient norm") from e

                        # Start with legacy-friendly keys so old dashboards work.
                        metrics: dict[str, float] = {
                            "loss": float(loss_val),
                            "train_loss": float(loss_val),
                            "ppl": float(ppl),
                            "train_ppl": float(ppl),
                            "lr": float(lr),
                            "lr_base": float(lr_base),
                            "lr_mult": float(lr_mult),
                            "grad_norm": float(grad_norm),
                            "grad_accum": float(getattr(train, "gradient_accumulation_steps", 1)),
                            "batch_size": float(getattr(train, "batch_size", 0)),
                            "seq_len": float(getattr(train, "block_size", 0)),
                        }
                        try:
                            # Best-effort: compute extra metrics on the last microbatch.
                            if outputs is not None and last_batch_td is not None:
                                extra = _call_objective_metrics(outputs=outputs, batch_td=last_batch_td, loss=loss)
                            else:
                                extra = None
                            if isinstance(extra, dict):
                                metrics.update({str(k): float(v) for k, v in extra.items()})
                        except Exception as e:
                            raise RuntimeError("Failed to compute objective metrics") from e

                        metrics.update(
                            {
                                "time_data_s": float(t_data - t_data0),
                                "time_fwd_s": float(t_fwd - t_data),
                                "time_bwd_s": float(t_bwd - t_fwd),
                                "time_optim_s": float(t_optim - t_bwd),
                                "time_step_s": float(t_optim - t0),
                                # Legacy-friendly ms fields.
                                "ms_fwd": float((t_fwd - t_data) * 1000.0),
                                "ms_bwd": float((t_bwd - t_fwd) * 1000.0),
                                "ms_opt": float((t_optim - t_bwd) * 1000.0),
                                "ms_step": float((t_optim - t0) * 1000.0),
                            }
                        )
                        # Token throughput (best-effort for token LM datasets).
                        try:
                            y = last_batch_td.get("target_ids", None) if last_batch_td is not None else None  # type: ignore[attr-defined]
                            if isinstance(y, Tensor):
                                step_s = float(max(1e-9, t_optim - t0))
                                metrics["tok_s"] = float(y.numel() * int(accum_steps)) / step_s
                        except Exception as e:
                            raise RuntimeError("Failed to compute token throughput") from e
                        # Memory footprint estimates (MiB).
                        try:
                            metrics.update(
                                {
                                    "mem_params_mb": bytes_to_mb(param_bytes()),
                                    "mem_grads_mb": bytes_to_mb(grad_bytes()),
                                    "mem_optim_mb": bytes_to_mb(optim_state_bytes()),
                                }
                            )
                        except Exception:
                            logger.warning("Failed to compute memory metrics (ignoring)")
                        if kernel_launches is not None:
                            metrics["kernel_events_estimate"] = float(kernel_launches)
                        metrics["compiled"] = 1.0 if compiled else 0.0

                        # MOSAIC memory stats (best-effort): include in JSONL + W&B.
                        try:
                            if isinstance(viz_ctx, TrainingVizMosaicContext) and viz_ctx.mosaic_mem_stats:
                                for k, v in viz_ctx.mosaic_mem_stats.items():
                                    if isinstance(v, (int, float)):
                                        metrics[str(k)] = float(v)
                                metrics["mosaic_teacher_p"] = float(getattr(viz_ctx, "mosaic_teacher_p", 1.0))

                                def _avg(suffix: str) -> float | None:
                                    vals = [
                                        float(v)
                                        for kk, v in viz_ctx.mosaic_mem_stats.items()
                                        if str(kk).endswith(suffix) and isinstance(v, (int, float))
                                    ]
                                    if not vals:
                                        return None
                                    return float(sum(vals) / float(len(vals)))

                                summary = {"teacher_p": float(getattr(viz_ctx, "mosaic_teacher_p", 1.0))}
                                for suf, key in [
                                    ("/read_hit_rate", "read_hit"),
                                    ("/read_hit_rate@req", "read_hit@req"),
                                    ("/read_conf@req", "read_conf@req"),
                                    ("/read_best_sim", "read_best_sim"),
                                    ("/read_conf", "read_conf"),
                                    ("/read_slot_entropy", "read_slot_ent"),
                                    ("/read_bucket_change_rate", "read_jitter"),
                                    ("/write_rate", "write_rate"),
                                    ("/write_update_frac", "write_update"),
                                    ("/write_insert_empty_frac", "write_insert_empty"),
                                    ("/write_evict_frac", "write_evict"),
                                    ("/write_bucket_full_frac", "write_full"),
                                    ("/write_gate_p_mean", "gate_p"),
                                    ("/write_gate_fire_frac", "gate_fire"),
                                    ("/write_bucket_entropy_norm", "write_ent"),
                                    ("/write_bucket_change_rate", "write_jitter"),
                                    ("/cand_buckets", "cand_buckets"),
                                    ("/drop_local_frac", "drop_local"),
                                    ("/rms_mem", "rms_mem"),
                                    ("/fuse_gate_mem_mean", "fuse_mem"),
                                    ("/rms_mem_contrib", "mem_contrib"),
                                ]:
                                    av = _avg(suf)
                                    if av is not None:
                                        summary[key] = av
                                logger.key_value(summary, title="MOSAIC memory stats (avg across layers)")
                        except Exception as e:
                            raise RuntimeError("Failed to log MOSAIC memory stats") from e
                        run_logger.log_metrics(
                            run_id=str(run.id),
                            phase="standard",
                            step=step + 1,
                            metrics=metrics,
                        )
                        if wandb_writer is not None:
                            # Log both: current platform namespace + legacy flat keys.
                            wandb_writer.log_scalars(prefix="train", step=step + 1, scalars=metrics)
                            wandb_writer.log_scalars(prefix="", step=step + 1, scalars=metrics)

                    progress.update(
                        task,
                        advance=1,
                        description=f"Step {step+1}/{run.steps} • loss={float(loss.detach()):.4f}",
                    )
        finally:
            if wandb_writer is not None:
                wandb_writer.close()
            # Cleanup hooks (best-effort).
            try:
                if "layer_stats_enabled" in locals() and layer_stats_enabled:
                    for h in _hook_handles:
                        try:
                            h.remove()
                        except Exception as e:
                            raise RuntimeError("Failed to remove hook") from e
            except Exception as e:
                raise RuntimeError("Failed to cleanup hooks") from e

        self._save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            run_id=str(run.id),
            phase="standard",
            step=int(run.steps),
            system=system,
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
        if not hasattr(dataset_comp, "build"):
            raise TypeError("Dataset component does not expose build()")
        dataset = dataset_comp.build()  # type: ignore[attr-defined]

        val_frac = float(defaults.data.val_frac)
        n = len(cast(Sized, dataset))
        n_train, _n_val = train_val_counts(n, float(val_frac))
        train_ds = Subset(dataset, range(0, n_train))

        def collate_to_tensordict(items: list[Any]) -> TensorDictBase:
            return collate_tensordict(items)

        loader_kwargs = {
            "batch_size": int(batch_size),
            "num_workers": int(train.num_workers),
            "pin_memory": bool(train.pin_memory) and device.type == "cuda",
            "drop_last": True,
            "collate_fn": collate_to_tensordict,
        }
        if dist_ctx is not None and hasattr(dist_ctx, "wrap_dataloader"):
            return dist_ctx.wrap_dataloader(train_ds, shuffle=True, **loader_kwargs)  # type: ignore[no-any-return, attr-defined]
        return DataLoader(train_ds, shuffle=True, **loader_kwargs)

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
        key = make_plan_key(payload)
        plan_path = checkpoint_dir / "plans" / f"{key}.json"

        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            return existing

        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            dtype_str = weight_dtype_str(device)

        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            amp_dtype_str = autocast_dtype_str(device)

        batch_size = int(train.batch_size)
        if bool(getattr(train, "auto_batch_size", False)):
            ref_block = int(getattr(train, "auto_batch_ref_block_size", 512))
            min_bs = int(getattr(train, "auto_batch_min", 1))
            batch_size = token_budget_batch_size(
                batch_size,
                block_size=int(train.block_size),
                ref_block_size=int(ref_block),
                min_batch_size=int(min_bs),
            )

        plan = RuntimePlan(
            key=key,
            device=str(device),
            torch_version=torch.__version__,
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=int(batch_size),
            compile=bool(getattr(train, "compile_model", False)),
            compile_mode=str(getattr(train, "compile_mode", "reduce-overhead")),
        )
        try:
            save_plan(plan_path, plan, payload=payload)
        except Exception as e:
            raise RuntimeError("Failed to save runtime plan") from e

        return plan
