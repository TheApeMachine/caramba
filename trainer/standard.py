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
from typing import Any, Protocol, Sized, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from carmath import (
    autocast_dtype,
    autocast_dtype_str,
    token_budget_batch_size,
    train_val_counts,
    weight_dtype_str,
)
from config.defaults import Defaults
from config.manifest import Manifest
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from console import logger
from instrumentation import RunLogger
from runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan
from runtime.tensordict_utils import TensorDictBase, as_tensordict, collate_tensordict, to_device


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
                from trainer.distributed import DistributedConfig, DistributedContext, DistributedStrategy

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
            except Exception:
                logger.warning("Failed to check if this is the main process (ignoring)")

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
                import torch.nn as nn

                m = getattr(system, "module", None)
                if isinstance(m, nn.Module):
                    system.module = dist_ctx.wrap_model(m)  # type: ignore[attr-defined]
            except Exception:
                logger.warning("Failed to wrap system module for DDP/FSDP (ignoring)")

        compiled = False
        if bool(getattr(runtime_plan, "compile", False)):
            # Best-effort torch.compile: only apply when a module is exposed.
            try:
                import torch.nn as nn

                m = getattr(system, "module", None)
                if isinstance(m, nn.Module):
                    system.module = torch.compile(m, mode=str(runtime_plan.compile_mode))  # type: ignore[attr-defined]
                    compiled = True
            except Exception:
                logger.warning("Failed to compile model (ignoring)")
                compiled = False

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
            from optimizer.lion import Lion

            optimizer = Lion(
                system.parameters(),  # type: ignore[arg-type]
                lr=float(train.lr),
                weight_decay=float(weight_decay),
                fused=bool(fused_opt),
            )
        else:
            raise ValueError(f"Unknown optimizer {opt_name!r}")

        from trainer.swap_manager import SwapManager

        swap = SwapManager(
            offload_optimizer=bool(getattr(train, "offload_optimizer", False)),
            offload_device="cpu",
        )

        use_amp = bool(train.use_amp)
        amp_dtype = autocast_dtype(device, str(train.amp_dtype))

        telemetry_interval = int(getattr(train, "telemetry_interval", 10)) or 10
        profile_every = int(getattr(train, "profile_every", 0)) or 0
        profile_record_shapes = bool(getattr(train, "profile_record_shapes", False))

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
            except Exception:
                return 0
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
        except Exception:
            logger.warning("Failed to emit telemetry (ignoring)")

        # Export a reproducibility artifact (lowered plan + io shapes).
        try:
            from compiler.plan import Planner

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
                    except Exception:
                        return out
                for k, v in items:
                    if isinstance(v, Tensor):
                        out[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                return out
            import json
            (checkpoint_dir / "io_shapes.json").write_text(
                json.dumps({"batch": shape_sig(b0), "outputs": shape_sig(o0)}, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to export IO shapes (ignoring)")

        logger.header("Training", f"{target.name}:{run.id} • {run.steps} steps")
        loader_iter = iter(loader)

        def _call_objective_loss(*, outputs: object, batch_td: TensorDictBase) -> Tensor:
            if not hasattr(objective, "loss"):
                raise TypeError("Objective component does not expose loss()")
            loss_fn = objective.loss  # type: ignore[attr-defined]
            try:
                sig = inspect.signature(loss_fn)
                params = sig.parameters
            except Exception:
                # If we can't introspect (e.g. some builtins), fall back to the canonical API.
                return loss_fn(outputs=outputs, batch=batch_td)

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
            except Exception:
                extra = metrics_fn(outputs=outputs, batch=batch_td, loss=loss)
                return cast(dict[str, float] | None, extra) if isinstance(extra, dict) else None

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
                outputs = system.forward(batch_td)  # type: ignore[attr-defined]
                loss = _call_objective_loss(outputs=outputs, batch_td=batch_td)
            return outputs, loss

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(run.steps)):
                t0 = time.perf_counter()
                try:
                    item = next(loader_iter)
                except StopIteration:
                    logger.warning("Reached end of loader, resetting")
                    loader_iter = iter(loader)
                    item = next(loader_iter)

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
                t_data = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                kernel_launches: int | None = None

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
                            outputs, loss = _forward_loss(batch_td)
                            loss.backward()
                        # Heuristic: count device-side events as “launch-ish”.
                        if device.type == "cuda":
                            try:
                                evs = prof.events() or []
                                # We intentionally keep this a heuristic; different builds expose
                                # different event metadata. The main goal is “fewer events over time”.
                                kernel_launches = int(len(evs))
                            except Exception:
                                logger.warning("Failed to count kernel launches (ignoring)")
                                kernel_launches = None
                    except Exception:
                        logger.warning("Failed to profile (ignoring)")
                        kernel_launches = None
                        outputs, loss = _forward_loss(batch_td)
                        loss.backward()
                else:
                    outputs, loss = _forward_loss(batch_td)

                t_fwd = time.perf_counter()
                # If we profiled, backward already happened.
                if not (profile_every > 0 and ((step + 1) % profile_every == 0)):
                    loss.backward()
                t_bwd = time.perf_counter()
                swap.before_optimizer_step(optimizer, device=device)
                optimizer.step()
                swap.after_optimizer_step(optimizer)
                if bool(getattr(train, "offload_optimizer", False)):
                    # After offloading, grads should be freed aggressively.
                    optimizer.zero_grad(set_to_none=True)
                t_optim = time.perf_counter()

                if (step + 1) % telemetry_interval == 0:
                    metrics: dict[str, float] = {"loss": float(loss.detach())}
                    try:
                        extra = _call_objective_metrics(outputs=outputs, batch_td=batch_td, loss=loss)
                        if isinstance(extra, dict):
                            metrics.update({str(k): float(v) for k, v in extra.items()})
                    except Exception:
                        # Metrics are best-effort; don't fail training.
                        logger.warning("Failed to compute objective metrics (ignoring)")
                    # Timing breakdown (seconds).
                    metrics.update(
                        {
                            "time_data_s": float(t_data - t0),
                            "time_fwd_s": float(t_fwd - t_data),
                            "time_bwd_s": float(t_bwd - t_fwd),
                            "time_optim_s": float(t_optim - t_bwd),
                            "time_step_s": float(t_optim - t0),
                        }
                    )
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
                    run_logger.log_metrics(
                        run_id=str(run.id),
                        phase="standard",
                        step=step + 1,
                        metrics=metrics,
                    )

                progress.update(
                    task,
                    advance=1,
                    description=f"Step {step+1}/{run.steps} • loss={float(loss.detach()):.4f}",
                )

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
        except Exception:
            logger.warning("Failed to save runtime plan (ignoring)")
        return plan
