from __future__ import annotations

from collections.abc import Iterator

import math
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config.run import Run
from console import logger
from carmath import autocast_dtype, safe_perplexity_from_nll
from trainer.collectors import Collector
from trainer.checkpointers import CheckPointer
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from trainer.upcycle_context import UpcycleContext
from runtime.tensordict_utils import TensorDictBase


def _sync_device(device: torch.device) -> None:
    """Synchronize for more accurate wall timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()  # type: ignore[attr-defined]


def _has_diffusion_head(student: nn.Module) -> bool:
    return hasattr(student, "diffusion_head") and getattr(student, "diffusion_head", None) is not None


def _get_diffusion_loss_weight(student: nn.Module) -> float:
    try:
        return float(student.config.diffusion_head.loss_weight)  # type: ignore[union-attr]
    except Exception:
        logger.warning("Failed to get diffusion loss weight; using default 0.10.")
        return 0.10


def _compute_loss(
    student: nn.Module,
    x: Tensor,
    y: Tensor,
    has_diffusion: bool,
    *,
    mps_fast_ce: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    def _tensor_stats(name: str, t: Tensor) -> str:
        try:
            t_f = t.detach().float()
            finite = torch.isfinite(t_f)
            n = int(t_f.numel())
            nf = int((~finite).sum().item())
            if n == 0:
                return f"{name}: empty"
            if nf == n:
                return f"{name}: all_nonfinite n={n} dtype={t.dtype} device={t.device}"
            tf = t_f[finite]
            return (
                f"{name}: dtype={t.dtype} device={t.device} shape={tuple(t.shape)} "
                f"nonfinite={nf}/{n} min={float(tf.min().item()):.5g} max={float(tf.max().item()):.5g} "
                f"mean={float(tf.mean().item()):.5g} std={float(tf.std(unbiased=False).item()):.5g}"
            )
        except Exception as e:
            return f"{name}: <stats_failed {e}>"

    if has_diffusion:
        result = student.forward(x, return_features=True)  # type: ignore[call-arg]
        features: Tensor = result[0]  # type: ignore[index]
        logits: Tensor = result[1]  # type: ignore[index]
        if not torch.isfinite(logits.detach()).all():
            raise RuntimeError(
                "Non-finite logits detected.\n"
                f"- {_tensor_stats('logits', logits)}\n"
                "This is a hard failure under the kernel policy (no silent fallback paths)."
            )
        # Default: compute CE in fp32 for numerical stability.
        # MPS fast path: avoid materializing fp32 logits for huge vocab (speed/memory win).
        use_fast = bool(mps_fast_ce) and logits.device.type == "mps"
        logits_ce = logits if use_fast else logits.float()
        ce_loss = F.cross_entropy(logits_ce.view(-1, logits_ce.shape[-1]), y.reshape(-1))
        diff_loss_val: Tensor = student.diffusion_loss(features, y)  # type: ignore[attr-defined]
        loss = ce_loss + _get_diffusion_loss_weight(student) * diff_loss_val
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(
                "Non-finite loss detected.\n"
                f"- {_tensor_stats('ce_loss', ce_loss)}\n"
                f"- {_tensor_stats('diff_loss', diff_loss_val)}\n"
                f"- {_tensor_stats('loss', loss)}"
            )
        return loss, ce_loss, diff_loss_val
    logits = student.forward(x)
    if not torch.isfinite(logits.detach()).all():
        raise RuntimeError(
            "Non-finite logits detected.\n"
            f"- {_tensor_stats('logits', logits)}\n"
            "This is a hard failure under the kernel policy (no silent fallback paths)."
        )
    use_fast = bool(mps_fast_ce) and logits.device.type == "mps"
    logits_ce = logits if use_fast else logits.float()
    ce_loss = F.cross_entropy(logits_ce.view(-1, logits_ce.shape[-1]), y.reshape(-1))
    if not torch.isfinite(ce_loss.detach()).all():
        raise RuntimeError(
            "Non-finite CE loss detected.\n"
            f"- {_tensor_stats('logits_ce', logits_ce)}\n"
            f"- {_tensor_stats('ce_loss', ce_loss)}"
        )
    return ce_loss, ce_loss, None


def _eval_global_loss(
    *,
    student: nn.Module,
    device: torch.device,
    dist_ctx: object | None,
    loader: DataLoader[TensorDictBase],
    max_batches: int = 2,
    mps_fast_ce: bool = False,
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
                use_fast = bool(mps_fast_ce) and logits.device.type == "mps"
                logits_ce = logits if use_fast else logits.float()
                ce = F.cross_entropy(logits_ce.view(-1, logits_ce.shape[-1]), y.reshape(-1))
                diff = student.diffusion_loss(features, y)  # type: ignore[attr-defined]
                loss = ce + float(diff_weight) * diff
                total_diff += float(diff)
            else:
                logits = student.forward(x)
                use_fast = bool(mps_fast_ce) and logits.device.type == "mps"
                logits_ce = logits if use_fast else logits.float()
                ce = F.cross_entropy(logits_ce.view(-1, logits_ce.shape[-1]), y.reshape(-1))
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
        logger.warning(f"Failed to convert {value} to int; using default {default}.")
        return int(default)


def _build_optimizer(train, params) -> Optimizer:
    """Build optimizer for upcycle steppers (blockwise/global)."""
    opt_name = str(getattr(train, "optimizer", "adamw")).lower()
    weight_decay = float(getattr(train, "weight_decay", 0.0))
    fused_opt = bool(getattr(train, "fused_optimizer", False))
    if opt_name in ("adamw", "adam"):
        return torch.optim.AdamW(params, lr=float(train.lr), weight_decay=float(weight_decay))
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=float(train.lr), weight_decay=float(weight_decay))
    if opt_name == "lion":
        from optimizer.lion import Lion

        return Lion(params, lr=float(train.lr), weight_decay=float(weight_decay), fused=bool(fused_opt))
    raise ValueError(f"Unknown optimizer {opt_name!r}")


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

        optimizer = _build_optimizer(train, ctx.student.parameters())
        ctx.student.train()

        has_diffusion = _has_diffusion_head(ctx.student)
        use_amp = bool(train.use_amp) and ctx.device.type in ("cuda", "mps", "cpu")
        amp_dtype = autocast_dtype(ctx.device, str(train.amp_dtype))

        scaler = None
        if use_amp and ctx.device.type == "cuda" and amp_dtype == torch.float16:
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                logger.warning("Failed to create GradScaler; disabling autocast.")
                scaler = None
        # Metal kernels on MPS currently require fp16/fp32 (bf16 is rejected).
        if use_amp and ctx.device.type == "mps" and amp_dtype == torch.bfloat16:
            logger.warning("Disabling bf16 autocast on MPS (Metal kernels require fp16/fp32); using fp32 math.")
            use_amp = False
        if use_amp and ctx.device.type == "mps" and scaler is None and amp_dtype == torch.float16:
            if bool(getattr(train, "mps_allow_fp16_autocast_without_gradscaler", False)):
                logger.warning("Enabling fp16 autocast on MPS without GradScaler (speed; may be unstable).")
            else:
                logger.warning("Disabling fp16 autocast on MPS (no GradScaler); using fp32 math for stability.")
                use_amp = False
        if ctx.device.type == "mps" and bool(getattr(train, "mps_force_fp32_weights", False)):
            try:
                if any(p.dtype in (torch.float16, torch.bfloat16) for p in ctx.student.parameters()):
                    logger.warning("Upcasting student weights to float32 on MPS for stability / Metal kernel compatibility.")
                    ctx.student.to(dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to upcast student to float32 on MPS: {e}")

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
                    logger.warning("Failed to load optimizer/scheduler state dict; continuing with defaults.")

        logger.header("Global Fine-tuning", f"{run.steps} steps")
        loader_iter = iter(loader)
        loss: Tensor | None = None
        accum_steps = int(getattr(train, "gradient_accumulation_steps", 1))
        accum_steps = max(1, accum_steps)
        best_val_loss = float("inf")
        best_val_ppl = float("inf")

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=run.steps)
            for step in range(int(start_step), int(run.steps)):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                ce_sum = 0.0
                diff_sum = 0.0
                diff_seen = False
                tokens_seen = 0
                fwd_s = 0.0
                bwd_s = 0.0
                t0 = time.perf_counter()

                autocast_enabled = bool(use_amp)
                for _micro in range(int(accum_steps)):
                    batch, loader_iter = collector.next_batch(loader, loader_iter)
                    x = batch["input_ids"].to(device=ctx.device)
                    y = batch["target_ids"].to(device=ctx.device)
                    tokens_seen += int(y.numel())
                    try:
                        with torch.autocast(
                            device_type=ctx.device.type,
                            dtype=amp_dtype,
                            enabled=autocast_enabled,
                        ):
                            tf0 = time.perf_counter()
                            loss, ce_loss, diff_loss = _compute_loss(
                                ctx.student,
                                x,
                                y,
                                has_diffusion,
                                mps_fast_ce=bool(getattr(train, "mps_fast_ce", False)),
                            )
                            tf1 = time.perf_counter()
                    except TypeError:
                        logger.warning("Failed to autocast loss computation; disabling autocast.")
                        autocast_enabled = False
                        tf0 = time.perf_counter()
                        loss, ce_loss, diff_loss = _compute_loss(
                            ctx.student,
                            x,
                            y,
                            has_diffusion,
                            mps_fast_ce=bool(getattr(train, "mps_fast_ce", False)),
                        )
                        tf1 = time.perf_counter()

                    # Accumulate gradients on scaled loss, but log unscaled averages.
                    loss_sum += float(loss.detach())
                    ce_sum += float(ce_loss.detach())
                    if diff_loss is not None:
                        diff_sum += float(diff_loss.detach())
                        diff_seen = True

                    scaled = loss / float(accum_steps)
                    tb0 = time.perf_counter()
                    if scaler is not None and autocast_enabled:
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    tb1 = time.perf_counter()
                    fwd_s += float(tf1 - tf0)
                    bwd_s += float(tb1 - tb0)

                grad_norm = 0.0
                grad_norm_post = 0.0
                grad_clip = float(getattr(train, "grad_clip_norm", 0.0))
                try:
                    from carmath import global_grad_norm_l2

                    grad_norm = float(global_grad_norm_l2(ctx.student))
                except Exception:
                    grad_norm = 0.0

                # Optional: clip to stabilize occasional spikes without changing LR.
                # Important: unscale before clipping when using GradScaler (CUDA fp16).
                if grad_clip > 0.0:
                    try:
                        if scaler is not None and autocast_enabled:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ctx.student.parameters(), max_norm=float(grad_clip))
                        try:
                            from carmath import global_grad_norm_l2

                            grad_norm_post = float(global_grad_norm_l2(ctx.student))
                        except Exception:
                            grad_norm_post = 0.0
                    except Exception:
                        grad_norm_post = 0.0
                else:
                    grad_norm_post = float(grad_norm)

                topt0 = time.perf_counter()
                if scaler is not None and autocast_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                _sync_device(ctx.device)
                topt1 = time.perf_counter()

                loss_val = float(loss_sum) / float(accum_steps)
                lr = float(optimizer.param_groups[0].get("lr", train.lr))
                lr_base = float(getattr(train, "lr", lr))
                lr_mult = (lr / lr_base) if lr_base > 0 else 1.0
                t1 = time.perf_counter()
                step_s = float(t1 - t0)
                tok_s = float(tokens_seen) / step_s if step_s > 0 else 0.0
                # Heuristic "gbs" throughput proxy (tokens * d_model * 2 / sec).
                d_model = float(getattr(getattr(ctx.student, "config", object()), "d_model", 0.0) or 0.0)
                if d_model <= 0.0:
                    d_model = float(getattr(train, "block_size", 0) or 0)
                gbs = (float(tokens_seen) * float(d_model) * 2.0) / (1e9 * step_s) if step_s > 0 else 0.0

                metrics: dict[str, float] = {
                    "loss": loss_val,
                    "ce_loss": float(ce_sum) / float(accum_steps),
                    "lr": lr,
                }
                if diff_seen:
                    metrics["diff_loss"] = float(diff_sum) / float(accum_steps)
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
                        vloss = float(val_metrics.get("val_loss", 0.0))
                        vppl = safe_perplexity_from_nll(vloss)
                        if vloss > 0.0 and vloss < best_val_loss:
                            best_val_loss = vloss
                        if vppl < best_val_ppl:
                            best_val_ppl = vppl
                        ctx.inst.log_scalars(
                            step=step + 1,
                            prefix="",
                            scalars={
                                "val_loss": float(vloss),
                                "val_ppl": float(vppl),
                                "best_val_loss": float(best_val_loss if math.isfinite(best_val_loss) else 0.0),
                                "best_val_ppl": float(best_val_ppl if math.isfinite(best_val_ppl) else 0.0),
                            },
                        )

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
