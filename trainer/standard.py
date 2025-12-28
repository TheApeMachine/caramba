"""Standard trainer (target-based).

This trainer is objective-driven:
- dataset provides batches
- system produces outputs
- objective computes loss from (batch, outputs)

No assumptions about "tokens" are baked into the trainer beyond the chosen
components.
"""
from __future__ import annotations

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
from trainer.objectives import LossBundle


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

        runtime_plan = self._load_or_create_runtime_plan(
            checkpoint_dir=checkpoint_dir,
            device=device,
            train=train,
            system_cfg=dict(target.system.config),
        )
        dtype = self._parse_dtype(runtime_plan.dtype)

        if hasattr(system, "to"):
            system.to(device=device, dtype=dtype)  # type: ignore[attr-defined]

        loader = self._build_loader(
            dataset_comp=dataset_comp,
            defaults=defaults,
            train=train,
            device=device,
            batch_size=int(runtime_plan.batch_size),
        )

        if not hasattr(system, "parameters"):
            raise TypeError("System component does not expose parameters()")
        optimizer = torch.optim.AdamW(system.parameters(), lr=train.lr)  # type: ignore[arg-type]

        use_amp = bool(train.use_amp)
        amp_dtype = autocast_dtype(device, str(train.amp_dtype))

        logger.header("Training", f"{target.name}:{run.id} • {run.steps} steps")
        loader_iter = iter(loader)

        def to_device(obj: Any) -> Any:
            if isinstance(obj, Tensor):
                return obj.to(device=device)
            if isinstance(obj, dict):
                return {k: to_device(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_device(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(to_device(v) for v in obj)
            return obj

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(run.steps)):
                try:
                    item = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    item = next(loader_iter)

                # Back-compat: datasets may yield (x, y) for LM.
                if isinstance(item, tuple) and len(item) == 2:
                    x, y = item
                    batch: dict[str, Any] = {"input_ids": x, "target_ids": y}
                elif isinstance(item, dict):
                    batch = dict(item)
                else:
                    batch = {"inputs": item}

                batch = cast(dict[str, Any], to_device(batch))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    if not hasattr(system, "forward"):
                        raise TypeError("System component does not expose forward()")
                    outputs = system.forward(batch)  # type: ignore[attr-defined]
                    if not hasattr(objective, "loss"):
                        raise TypeError("Objective component does not expose loss()")
                    loss_bundle: LossBundle = objective.loss(  # type: ignore[attr-defined]
                        outputs=outputs, batch=batch
                    )

                loss_bundle.total.backward()
                optimizer.step()

                if (step + 1) % 10 == 0:
                    metrics = {"loss": float(loss_bundle.total), **loss_bundle.parts}
                    run_logger.log_metrics(
                        run_id=str(run.id),
                        phase="standard",
                        step=step + 1,
                        metrics=metrics,
                    )

                progress.update(
                    task,
                    advance=1,
                    description=f"Step {step+1}/{run.steps} • loss={float(loss_bundle.total):.4f}",
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
    ) -> DataLoader:
        if not hasattr(dataset_comp, "build"):
            raise TypeError("Dataset component does not expose build()")
        dataset = dataset_comp.build()  # type: ignore[attr-defined]

        val_frac = float(defaults.data.val_frac)
        n = len(cast(Sized, dataset))
        n_train, _n_val = train_val_counts(n, float(val_frac))
        train_ds = Subset(dataset, range(0, n_train))

        loader_kwargs = {
            "batch_size": int(batch_size),
            "num_workers": int(train.num_workers),
            "pin_memory": bool(train.pin_memory) and device.type == "cuda",
            "drop_last": True,
        }
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
