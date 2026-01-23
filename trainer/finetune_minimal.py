"""Minimal finetuning trainer.

Focused on straightforward LM finetuning: dataset → forward → loss → step.
No extra instrumentation, no special research features.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from carmath import autocast_dtype, global_grad_norm_l2, safe_perplexity_from_nll, token_budget_batch_size, weight_dtype
from config.defaults import Defaults
from config.manifest import Manifest
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from console import logger
from instrumentation import RunLogger
from instrumentation.wandb_writer import WandBWriter
from runtime.tensordict_utils import TensorDictBase, as_tensordict, to_device
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from trainer.train_dataloader.builder import TrainDataLoaderBuilder
from optimizer.adamw_master import AdamWMaster
from optimizer.lion import Lion


class _Engine(Protocol):
    registry: Any


class FinetuneMinimalTrainer:
    def __init__(self, *, checkpoint_dir: str | None = None) -> None:
        self._checkpoint_dir_override = checkpoint_dir
        self._train_dataloader_builder = TrainDataLoaderBuilder()

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
        if not hasattr(system, "forward") or not callable(getattr(system, "forward")):
            raise TypeError("System component does not expose a callable forward()")

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
                    f"trainer.finetune_minimal only supports phase=standard, got {run.train.phase}"
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
        logger.info(f"Running finetune_minimal for {target.name}:{run.id} with device={train.device} dtype={train.dtype}")
        device = torch.device(train.device)
        dtype = weight_dtype(device, str(train.dtype))

        if hasattr(system, "to"):
            system.to(device=device, dtype=dtype)  # type: ignore[attr-defined]
        if hasattr(system, "train"):
            system.train(True)  # type: ignore[attr-defined]

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

        loader = self._train_dataloader_builder.build(
            dataset_comp=dataset_comp,
            defaults=defaults,
            train=train,
            device=device,
            batch_size=int(batch_size),
            dist_ctx=None,
        )

        optimizer = self._build_optimizer(system=system, train=train, device=device, dtype=dtype)
        lr_scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        use_amp = bool(getattr(train, "use_amp", False))
        amp_dtype = autocast_dtype(device, str(getattr(train, "amp_dtype", "float16")))
        scaler = None
        if use_amp and device.type == "cuda" and amp_dtype == torch.float16:
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                logger.warning("Failed to create GradScaler; disabling autocast for stability.")
                scaler = None
                use_amp = False
        if use_amp and device.type == "mps" and amp_dtype == torch.bfloat16:
            logger.warning("Disabling bf16 autocast on MPS; using fp32 math for stability.")
            use_amp = False
        if use_amp and device.type == "mps" and scaler is None and amp_dtype == torch.float16:
            logger.warning("Disabling fp16 autocast on MPS (no GradScaler); using fp32 math.")
            use_amp = False

        wandb_writer: WandBWriter | None = None
        if bool(getattr(defaults.logging, "wandb", False)):
            try:
                wandb_writer = WandBWriter(
                    out_dir=checkpoint_dir / "wandb" / str(run.id),
                    enabled=True,
                    project=str(getattr(defaults.logging, "wandb_project", "")),
                    entity=str(getattr(defaults.logging, "wandb_entity", "") or "") or None,
                    mode=str(getattr(defaults.logging, "wandb_mode", "online")),
                    run_name=f"{target.name}:{run.id}",
                    group=str(target.name),
                    tags=["finetune_minimal"],
                    config={
                        "trainer": "finetune_minimal",
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

        if not hasattr(objective, "loss"):
            raise TypeError("Objective component does not expose loss()")
        loss_fn = objective.loss  # type: ignore[attr-defined]
        loss_batch_key = self._resolve_loss_batch_key(loss_fn)

        telemetry_interval = int(getattr(train, "telemetry_interval", 10) or 10)
        telemetry_interval = max(1, telemetry_interval)
        accum_steps = max(1, int(getattr(train, "gradient_accumulation_steps", 1) or 1))
        grad_clip = float(getattr(train, "grad_clip_norm", 0.0) or 0.0)

        logger.header("Training", f"{target.name}:{run.id} • {run.steps} steps (finetune_minimal)")
        loader_iter = iter(loader)

        for step0 in range(int(run.steps)):
            step_1 = int(step0 + 1)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0

            for _ in range(int(accum_steps)):
                item, loader_iter = self._next_loader_item(loader, loader_iter)
                if isinstance(item, TensorDictBase):
                    batch_td = item
                elif isinstance(item, dict):
                    batch_td = as_tensordict(item)
                else:
                    raise TypeError(
                        "FinetuneMinimalTrainer expects batch items to be dict/TensorDict. "
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

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = system.forward(mb, ctx=None)  # type: ignore[attr-defined]
                    loss = self._call_loss(loss_fn, loss_batch_key, outputs=outputs, batch_td=mb)

                loss_sum += float(loss.detach())
                scaled = loss / float(accum_steps)
                if scaler is not None and use_amp:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if grad_clip > 0.0:
                if scaler is not None and use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                    system.parameters(),  # type: ignore[arg-type]
                    max_norm=float(grad_clip),
                )

            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if (step_1 % telemetry_interval) == 0:
                loss_val = float(loss_sum) / float(accum_steps)
                ppl = float(safe_perplexity_from_nll(float(loss_val)))
                lrs = [float(g.get("lr", float(train.lr))) for g in optimizer.param_groups]
                lr_min = float(min(lrs)) if lrs else float(train.lr)
                lr_max = float(max(lrs)) if lrs else float(train.lr)
                grad_norm = 0.0
                if grad_clip > 0.0:
                    try:
                        grad_norm = float(global_grad_norm_l2(system))  # type: ignore[arg-type]
                    except Exception:
                        grad_norm = 0.0

                metrics = {
                    "loss": loss_val,
                    "train_loss": loss_val,
                    "ppl": ppl,
                    "train_ppl": ppl,
                    "lr": lr_max,
                    "lr_min": lr_min,
                    "lr_max": lr_max,
                    "grad_norm": grad_norm,
                    "grad_accum": float(accum_steps),
                    "batch_size": float(batch_size),
                    "seq_len": float(getattr(train, "block_size", 0)),
                }
                run_logger.log_event(
                    type="metrics",
                    run_id=str(run.id),
                    phase="standard",
                    step=int(step_1),
                    data=metrics,
                )
                if wandb_writer is not None:
                    wandb_writer.log_scalars(prefix="metrics", step=int(step_1), scalars=metrics)

        self._save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            run_id=str(run.id),
            phase="standard",
            step=int(run.steps),
            system=system,
        )

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
                    logger.warning(
                        "AdamW fused optimizer requested but unsupported; falling back to torch.optim.AdamW. "
                        f"device={device.type} dtype={dtype}"
                    )
                    fused_opt = False
                logger.info(f"optimizer=adamw fused=false backend=torch_adamw device={device.type} dtype={dtype}")
                return torch.optim.AdamW(
                    params,  # type: ignore[arg-type]
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=float(weight_decay),
                )

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
            return "batch_td"

        raise TypeError("Objective.loss must accept a batch keyword (e.g. 'batch' or '_batch')")

    def _call_loss(
        self,
        loss_fn: Any,
        loss_batch_key: str,
        *,
        outputs: object,
        batch_td: TensorDictBase,
    ) -> Tensor:
        if loss_batch_key == "batch":
            loss = loss_fn(outputs=outputs, batch=batch_td)
        elif loss_batch_key == "_batch":
            loss = loss_fn(outputs=outputs, _batch=batch_td)
        elif loss_batch_key == "batch_td":
            loss = loss_fn(outputs=outputs, batch_td=batch_td)
        else:
            loss = loss_fn(outputs=outputs, batch=batch_td, batch_td=batch_td)
        if not isinstance(loss, Tensor):
            raise TypeError(f"Objective.loss must return a Tensor, got {type(loss).__name__}")
        return loss

    def _next_loader_item(self, loader: Any, it: Any) -> tuple[Any, Any]:
        try:
            item = next(it)
            return item, it
        except StopIteration:
            it2 = iter(loader)
            try:
                item = next(it2)
            except StopIteration as e:
                raise RuntimeError("Dataloader is empty; cannot fetch a batch") from e
            return item, it2

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
