from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from config.run import Run
from console import logger
from carmath import autocast_dtype
from trainer.collectors import Collector
from trainer.checkpointers import CheckPointer
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from trainer.upcycle_context import UpcycleContext
from runtime.tensordict_utils import TensorDictBase


def _has_diffusion_head(student: nn.Module) -> bool:
    return hasattr(student, "diffusion_head") and getattr(student, "diffusion_head", None) is not None


def _get_diffusion_loss_weight(student: nn.Module) -> float:
    try:
        return float(student.config.diffusion_head.loss_weight)  # type: ignore[union-attr]
    except Exception:
        return 0.10


def _compute_loss(
    student: nn.Module, x: Tensor, y: Tensor, has_diffusion: bool
) -> tuple[Tensor, Tensor, Tensor | None]:
    if has_diffusion:
        result = student.forward(x, return_features=True)  # type: ignore[call-arg]
        features: Tensor = result[0]  # type: ignore[index]
        logits: Tensor = result[1]  # type: ignore[index]
        ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        diff_loss_val: Tensor = student.diffusion_loss(features, y)  # type: ignore[attr-defined]
        loss = ce_loss + _get_diffusion_loss_weight(student) * diff_loss_val
        return loss, ce_loss, diff_loss_val
    logits = student.forward(x)
    ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
    return ce_loss, ce_loss, None


def _eval_global_loss(
    *,
    student: nn.Module,
    device: torch.device,
    dist_ctx: object | None,
    loader: DataLoader[TensorDictBase],
    max_batches: int = 2,
) -> dict[str, float]:
    was_training = student.training
    student.eval()

    has_diffusion = _has_diffusion_head(student)
    diff_weight = _get_diffusion_loss_weight(student) if has_diffusion else 0.0

    total_loss = 0.0
    total_ce = 0.0
    total_diff = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device=device)
            y = batch["target_ids"].to(device=device)
            if has_diffusion:
                result = student.forward(x, return_features=True)  # type: ignore[call-arg]
                features: Tensor = result[0]  # type: ignore[index]
                logits: Tensor = result[1]  # type: ignore[index]
                ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
                diff = student.diffusion_loss(features, y)  # type: ignore[attr-defined]
                loss = ce + float(diff_weight) * diff
                total_diff += float(diff)
            else:
                logits = student.forward(x)
                ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
                loss = ce
            total_loss += float(loss)
            total_ce += float(ce)
            n += 1
            if n >= int(max_batches):
                break

    if dist_ctx is not None and n > 0:
        # type: ignore[attr-defined]
        t = torch.tensor([total_loss, total_ce, total_diff, float(n)], device=device)
        t = dist_ctx.all_reduce(t, op="sum")  # type: ignore[attr-defined]
        total_loss = float(t[0].item())
        total_ce = float(t[1].item())
        total_diff = float(t[2].item())
        n = int(t[3].item())

    if was_training:
        student.train()

    denom = float(n) if n > 0 else 1.0
    metrics: dict[str, float] = {"val_loss": total_loss / denom, "val_ce_loss": total_ce / denom}
    if has_diffusion:
        metrics["val_diff_loss"] = total_diff / denom
    return metrics


def _int_or(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


class GlobalStepper:
    def run(
        self,
        run: Run,
        ctx: UpcycleContext,
        *,
        collector: Collector,
        checkpointer: CheckPointer,
        save_every: int,
        resume_state: dict[str, object] | None,
    ) -> None:
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        train = run.train

        loader, val_loader = collector.build_loaders(train, ctx)
        for p in ctx.student.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(ctx.student.parameters(), lr=train.lr)
        ctx.student.train()

        has_diffusion = _has_diffusion_head(ctx.student)
        use_amp = bool(train.use_amp) and ctx.device.type in ("cuda", "mps", "cpu")
        amp_dtype = autocast_dtype(ctx.device, str(train.amp_dtype))

        scaler = None
        if use_amp and ctx.device.type == "cuda" and amp_dtype == torch.float16:
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                scaler = None

        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        start_step = 0
        if resume_state is not None and str(resume_state.get("phase", "")) == "global":
            if str(resume_state.get("run_id", "")) == str(run.id):
                start_step = _int_or(resume_state.get("step"), 0)
                try:
                    if "optimizer_state_dict" in resume_state:
                        optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
                    if scheduler is not None and "scheduler_state_dict" in resume_state:
                        scheduler.load_state_dict(resume_state["scheduler_state_dict"])  # type: ignore[arg-type]
                except Exception:
                    pass

        logger.header("Global Fine-tuning", f"{run.steps} steps")
        loader_iter = iter(loader)
        loss: Tensor | None = None

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=run.steps)
            for step in range(int(start_step), int(run.steps)):
                batch, loader_iter = collector.next_batch(loader, loader_iter)
                x = batch["input_ids"].to(device=ctx.device)
                y = batch["target_ids"].to(device=ctx.device)

                optimizer.zero_grad(set_to_none=True)
                autocast_enabled = bool(use_amp)
                try:
                    with torch.autocast(
                        device_type=ctx.device.type,
                        dtype=amp_dtype,
                        enabled=autocast_enabled,
                    ):
                        loss, ce_loss, diff_loss = _compute_loss(ctx.student, x, y, has_diffusion)
                except TypeError:
                    autocast_enabled = False
                    loss, ce_loss, diff_loss = _compute_loss(ctx.student, x, y, has_diffusion)

                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss) if loss is not None else 0.0
                lr = float(optimizer.param_groups[0].get("lr", train.lr))
                metrics: dict[str, float] = {"loss": loss_val, "ce_loss": float(ce_loss), "lr": lr}
                if diff_loss is not None:
                    metrics["diff_loss"] = float(diff_loss)
                if ctx.inst:
                    ctx.inst.log_scalars(step=step + 1, prefix="train/global", scalars=metrics)

                if (
                    val_loader is not None
                    and ctx.defaults is not None
                    and int(getattr(ctx.defaults, "eval_iters", 0)) > 0
                    and ((step + 1) % int(getattr(ctx.defaults, "eval_iters", 0)) == 0)
                ):
                    val_metrics = _eval_global_loss(
                        student=ctx.student,
                        device=ctx.device,
                        dist_ctx=ctx.dist_ctx,
                        loader=val_loader,
                        max_batches=2,
                    )
                    if ctx.inst:
                        ctx.inst.log_scalars(step=step + 1, prefix="eval/global", scalars=val_metrics)

                desc = f"Step {step + 1}/{run.steps} • loss={loss_val:.4f}"
                progress.update(task, advance=1, description=desc)

                current_step = step + 1
                if save_every > 0 and current_step % int(save_every) == 0:
                    checkpointer.save(
                        ctx=ctx,
                        run_id=str(run.id),
                        phase="global",
                        step=int(current_step),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=int(current_step),
                    )

        if loss is not None and ctx.dist_ctx is not None:
            loss = ctx.dist_ctx.all_reduce(loss.detach(), op="avg")  # type: ignore[assignment]
        if loss is not None:
            logger.success(f"Global fine-tuning complete • final loss={float(loss):.6f}")

        checkpointer.save(
            ctx=ctx,
            run_id=str(run.id),
            phase="global",
            step=int(run.steps),
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=int(run.steps),
            is_final=True,
        )

