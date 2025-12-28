"""Gradient isolation trainer (trainable scopes).

This trainer generalizes blockwise/adapters-only training by allowing a manifest
to define which parameters are trainable via regex patterns over parameter names.

It reuses the StandardTrainer loop but freezes parameters outside the scope.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
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


def apply_trainable_scope(
    system: object,
    *,
    trainable: list[str],
    frozen: list[str] | None = None,
) -> dict[str, int]:
    """Freeze/unfreeze parameters by regex over `named_parameters()`."""
    if not hasattr(system, "named_parameters"):
        raise TypeError("System must expose named_parameters() for trainable_scope")

    train_re = [re.compile(p) for p in trainable]
    frozen_re = [re.compile(p) for p in (frozen or [])]

    total = 0
    trainable_n = 0
    for name, p in system.named_parameters():  # type: ignore[attr-defined]
        if not isinstance(p, Tensor):
            continue
        total += 1
        # Frozen overrides trainable if both match.
        if any(r.search(str(name)) for r in frozen_re):
            p.requires_grad = False
            continue
        if any(r.search(str(name)) for r in train_re):
            p.requires_grad = True
            trainable_n += 1
        else:
            p.requires_grad = False

    if trainable_n <= 0:
        raise ValueError(f"trainable_scope matched 0 parameters out of {total}")
    return {"total": int(total), "trainable": int(trainable_n)}


class GradientIsolationTrainer:
    def __init__(
        self,
        *,
        trainable_scope: str | list[str],
        frozen_scope: str | list[str] | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.trainable_scope = (
            [str(trainable_scope)]
            if isinstance(trainable_scope, str)
            else [str(x) for x in trainable_scope]
        )
        self.frozen_scope = (
            None
            if frozen_scope is None
            else ([str(frozen_scope)] if isinstance(frozen_scope, str) else [str(x) for x in frozen_scope])
        )
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
            return None

        dataset_comp = engine.registry.build(target.data, backend=str(target.backend))
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
                    f"trainer.gradient_isolation only supports phase=standard, got {run.train.phase}"
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
                pass

        # Apply scope after moving to device (so params exist and are typed).
        scope_stats = apply_trainable_scope(
            system,
            trainable=list(self.trainable_scope),
            frozen=None if self.frozen_scope is None else list(self.frozen_scope),
        )
        run_logger.log_event(
            type="trainable_scope",
            run_id=str(run.id),
            phase="standard",
            step=0,
            data={"trainable_scope": self.trainable_scope, "frozen_scope": self.frozen_scope, **scope_stats},
        )

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

        logger.header(
            "Training (GradientIsolation)",
            f"{target.name}:{run.id} • {run.steps} steps • trainable={scope_stats['trainable']}/{scope_stats['total']}",
        )
        loader_iter = iter(loader)

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(run.steps)):
                t0 = time.perf_counter()
                try:
                    item = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    item = next(loader_iter)

                if isinstance(item, TensorDictBase):
                    batch_td = item
                elif isinstance(item, dict):
                    batch_td = as_tensordict(item)
                else:
                    raise TypeError(f"Expected dict/TensorDict batch, got {type(item).__name__}")

                batch_td = cast(TensorDictBase, to_device(batch_td, device=device))
                t_data = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    if not hasattr(system, "forward"):
                        raise TypeError("System component does not expose forward()")
                    outputs = system.forward(batch_td)  # type: ignore[attr-defined]
                    if not hasattr(objective, "loss"):
                        raise TypeError("Objective component does not expose loss()")
                    loss = objective.loss(outputs=outputs, batch=batch_td)  # type: ignore[attr-defined]
                    if not isinstance(loss, Tensor):
                        raise TypeError(
                            f"Objective.loss must return a Tensor, got {type(loss).__name__}"
                        )
                t_fwd = time.perf_counter()
                loss.backward()
                t_bwd = time.perf_counter()
                swap.before_optimizer_step(optimizer, device=device)
                optimizer.step()
                swap.after_optimizer_step(optimizer)
                if bool(getattr(train, "offload_optimizer", False)):
                    optimizer.zero_grad(set_to_none=True)
                t_optim = time.perf_counter()

                if (step + 1) % telemetry_interval == 0:
                    metrics: dict[str, float] = {"loss": float(loss.detach())}
                    if hasattr(objective, "metrics"):
                        try:
                            extra = objective.metrics(  # type: ignore[attr-defined]
                                outputs=outputs, batch=batch_td, loss=loss
                            )
                            if isinstance(extra, dict):
                                metrics.update({str(k): float(v) for k, v in extra.items()})
                        except Exception:
                            pass
                    metrics.update(
                        {
                            "time_data_s": float(t_data - t0),
                            "time_fwd_s": float(t_fwd - t_data),
                            "time_bwd_s": float(t_bwd - t_fwd),
                            "time_optim_s": float(t_optim - t_bwd),
                            "time_step_s": float(t_optim - t0),
                        }
                    )
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
        train_ds = Subset(dataset, range(n_train))

        loader_kwargs = {
            "batch_size": int(batch_size),
            "num_workers": int(train.num_workers),
            "pin_memory": bool(train.pin_memory) and device.type == "cuda",
            "drop_last": True,
            "collate_fn": collate_tensordict,
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
            pass
        return plan

