from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from caramba.config.run import Run
from caramba.config.train import TrainConfig
from caramba.console import logger
from caramba.layer.attention import AttentionLayer
from caramba.orchestrator.telemetry import SpikeDetector
from caramba.trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
from caramba.trainer.collectors import Collector
from caramba.trainer.checkpointers import CheckPointer
from caramba.trainer.distill import DistillLoss
from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from caramba.trainer.upcycle_context import UpcycleContext
from caramba.carmath import autocast_dtype
from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.topology.residual import ResidualTopology


def _int_or(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


def _build_blockwise_config(train: TrainConfig, device: torch.device) -> BlockwiseConfig:
    return BlockwiseConfig(
        cache_teacher_outputs=bool(train.cache_teacher_outputs),
        max_cache_size=int(getattr(train, "blockwise_teacher_cache_max_size", 100)),
        use_amp=bool(train.use_amp),
        amp_dtype=autocast_dtype(device, str(train.amp_dtype)),
        accumulation_steps=int(train.gradient_accumulation_steps),
        use_truncated_forward=bool(getattr(train, "blockwise_truncated_forward", True)),
        grad_clip_norm=float(getattr(train, "blockwise_grad_clip_norm", 0.0)),
    )


def _build_optimizer(train: TrainConfig, params) -> Optimizer:
    """Build optimizer for upcycle steppers (blockwise/global)."""
    opt_name = str(getattr(train, "optimizer", "adamw")).lower()
    weight_decay = float(getattr(train, "weight_decay", 0.0))
    fused_opt = bool(getattr(train, "fused_optimizer", False))
    if opt_name in ("adamw", "adam"):
        return torch.optim.AdamW(
            params,
            lr=float(train.lr),
            weight_decay=float(weight_decay),
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(train.lr),
            weight_decay=float(weight_decay),
        )
    if opt_name == "lion":
        from caramba.optimizer.lion import Lion

        return Lion(
            params,
            lr=float(train.lr),
            weight_decay=float(weight_decay),
            fused=bool(fused_opt),
        )
    raise ValueError(f"Unknown optimizer {opt_name!r}")


def _scale_lr(
    *,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    factor: float,
    min_lr: float,
) -> float:
    """Scale optimizer LR (and scheduler base_lrs if present).

    LambdaLR overwrites param_group['lr'] from scheduler.base_lrs each step, so
    adaptive LR must update base_lrs too.
    """
    f = float(factor)
    if f <= 0:
        return float(optimizer.param_groups[0].get("lr", 0.0))

    for i, g in enumerate(optimizer.param_groups):
        cur = float(g.get("lr", 0.0))
        nxt = max(float(min_lr), cur * f)
        g["lr"] = nxt

        if scheduler is not None and i < len(scheduler.base_lrs):
            scheduler.base_lrs[i] = max(float(min_lr), float(scheduler.base_lrs[i]) * f)

    return float(optimizer.param_groups[0].get("lr", 0.0))


def _autotune_mode(train: TrainConfig) -> str:
    enabled = bool(getattr(train, "blockwise_autotune_enabled", False))
    mode = str(getattr(train, "blockwise_autotune_mode", "monitor")).lower().strip()
    if not enabled:
        return "off"
    if mode in {"off", "disabled"}:
        return "off"
    if mode in {"active", "monitor"}:
        return mode
    return "monitor"


class BlockwiseStepper:
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

        loader, _ = collector.build_loaders(train, ctx)
        optimizer = _build_optimizer(train, ctx.student.parameters())

        distill_target = str(getattr(train, "blockwise_distill_target", "attention")).lower().strip()
        trace_predicate: Callable[[str, torch.nn.Module], bool] | None = None
        if distill_target in {"residual", "post_residual", "post-residual"}:
            # Trace the *post-residual* output of the attention residual block.
            # This often produces a more stable signal for deeper layers since it
            # matches what is actually fed into the next layer.
            def _trace_predicate(name: str, m: torch.nn.Module) -> bool:
                if not isinstance(m, ResidualTopology):
                    return False
                try:
                    layers = getattr(m, "layers", None)
                    if layers is None:
                        return False
                    return any(isinstance(x, AttentionLayer) for x in layers)
                except Exception:
                    return any(isinstance(x, AttentionLayer) for x in m.modules())
            trace_predicate = _trace_predicate

        trainer = BlockwiseTrainer(
            teacher=ctx.teacher,
            student=ctx.student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _, m: isinstance(m, AttentionLayer),
            trace_predicate=trace_predicate,
            config=_build_blockwise_config(train, ctx.device),
        )
        n_blocks = trainer.block_count()

        target_loss: float | None = None
        if train.convergence_target is not None:
            target_loss = float(train.convergence_target)
        convergence_mode = target_loss is not None
        steps_per_block = (
            int(getattr(train, "convergence_max_steps", 0)) if convergence_mode else int(run.steps)
        )
        steps_per_block = max(1, int(steps_per_block))

        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                # In convergence mode we don't know the exact number of steps in advance,
                # so we conservatively budget max_steps per block.
                total_steps=int(n_blocks * steps_per_block),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )
        ctx.student.train()
        loader_iter = iter(loader)

        resume_block_index = 0
        resume_block_step = 0
        resume_global_step = 0
        if resume_state is not None and str(resume_state.get("phase", "")) == "blockwise":
            if str(resume_state.get("run_id", "")) == str(run.id):
                resume_block_index = _int_or(resume_state.get("block_index"), 0)
                resume_block_step = _int_or(resume_state.get("block_step"), 0)
                resume_global_step = _int_or(resume_state.get("global_step"), 0)
                try:
                    if "optimizer_state_dict" in resume_state:
                        optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
                    if scheduler is not None and "scheduler_state_dict" in resume_state:
                        scheduler.load_state_dict(resume_state["scheduler_state_dict"])  # type: ignore[arg-type]
                except Exception:
                    pass

        self._run_blocks(
            run_id=str(run.id),
            train=train,
            trainer=trainer,
            loader=loader,
            loader_iter=loader_iter,
            n_blocks=n_blocks,
            steps_per_block=int(steps_per_block),
            lr_scheduler=scheduler,
            ctx=ctx,
            collector=collector,
            checkpointer=checkpointer,
            save_every=int(save_every),
            start_block_index=resume_block_index,
            start_block_step=resume_block_step,
            start_global_step=resume_global_step,
            target_loss=target_loss,
            patience=(int(getattr(train, "convergence_patience", 0)) if convergence_mode else 0),
        )

        checkpointer.save(
            ctx=ctx,
            run_id=str(run.id),
            phase="blockwise",
            step=0,
            optimizer=optimizer,
            scheduler=scheduler,
            is_final=True,
        )

    def _run_blocks(
        self,
        *,
        run_id: str,
        train: TrainConfig,
        trainer: BlockwiseTrainer,
        loader: DataLoader[TensorDictBase],
        loader_iter: Iterator[TensorDictBase],
        n_blocks: int,
        steps_per_block: int,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
        ctx: UpcycleContext,
        collector: Collector,
        checkpointer: CheckPointer,
        save_every: int,
        start_block_index: int = 0,
        start_block_step: int = 0,
        start_global_step: int = 0,
        target_loss: float | None = None,
        patience: int = 0,
    ) -> None:
        convergence_mode = target_loss is not None
        total_steps = int(n_blocks * int(steps_per_block))
        global_step = int(start_global_step)

        mode = _autotune_mode(train)
        autotune_min_lr = float(getattr(train, "blockwise_autotune_min_lr", 1e-6))
        autotune_decay = float(getattr(train, "blockwise_autotune_lr_decay", 0.5))
        autotune_patience = int(getattr(train, "blockwise_autotune_plateau_patience", 100))
        autotune_log_every = int(getattr(train, "blockwise_autotune_log_every", 50))
        autotune_ema = float(getattr(train, "blockwise_autotune_ema_decay", 0.99))
        autotune_spike_std = float(getattr(train, "blockwise_autotune_spike_std", 3.0))
        autotune_window = int(getattr(train, "blockwise_autotune_window_size", 50))

        progress_ctx = logger.progress_bar() if not convergence_mode else nullcontext()
        with progress_ctx as progress:
            overall_task = None
            if progress is not None:
                overall_task = progress.add_task(f"Training {n_blocks} blocks...", total=total_steps)

            for block_index in range(int(start_block_index), n_blocks):
                loss = None
                best_loss = float("inf")
                steps_wo_improve = 0
                detector = SpikeDetector(
                    ema_decay=float(autotune_ema),
                    threshold_std=float(autotune_spike_std),
                    window_size=int(autotune_window),
                )

                mode_label = "(convergence)" if convergence_mode else ""
                logger.subheader(f"Blockwise{mode_label} • block={block_index + 1}/{n_blocks}")

                # Optional: reset LR at each new block so autotune decisions from prior
                # blocks don't silently starve later blocks.
                if bool(getattr(train, "blockwise_reset_lr_each_block", False)):
                    base_lr = float(getattr(train, "lr", 0.0))
                    for i, g in enumerate(trainer.optimizer.param_groups):
                        g["lr"] = base_lr
                        if lr_scheduler is not None and i < len(lr_scheduler.base_lrs):
                            lr_scheduler.base_lrs[i] = base_lr

                lr0 = float(trainer.optimizer.param_groups[0].get("lr", 0.0))
                header_kv: dict[str, str] = {
                    "optimizer": str(getattr(train, "optimizer", "adamw")),
                    "scheduler": str(getattr(train, "scheduler", "none")),
                    "lr": f"{lr0:.3e}",
                    "autotune_mode": mode,
                    "distill_target": str(getattr(train, "blockwise_distill_target", "attention")),
                    "cache_teacher_outputs": str(bool(getattr(train, "cache_teacher_outputs", True))),
                    "teacher_cache_max": str(int(getattr(train, "blockwise_teacher_cache_max_size", 100))),
                    "use_amp": str(bool(getattr(train, "use_amp", False))),
                    "truncated_forward": str(bool(getattr(train, "blockwise_truncated_forward", True))),
                    "grad_clip_norm": f"{float(getattr(train, 'blockwise_grad_clip_norm', 0.0)):.3g}",
                }
                if convergence_mode:
                    header_kv["target_loss"] = f"{float(target_loss):.6f}"
                    header_kv["patience"] = str(int(patience))
                    header_kv["max_steps"] = str(int(steps_per_block))
                logger.key_value(header_kv)

                step0 = int(start_block_step) if block_index == int(start_block_index) else 0
                block_step_end = int(steps_per_block)
                for step in range(step0, int(steps_per_block)):
                    batch, loader_iter = collector.next_batch(loader, loader_iter)
                    x = batch["input_ids"].to(device=ctx.device)
                    loss = trainer.step(x, block_index=block_index)
                    global_step += 1

                    loss_val = float(loss) if loss is not None else 0.0
                    if loss_val < best_loss - 1e-12:
                        best_loss = loss_val
                        steps_wo_improve = 0
                    else:
                        steps_wo_improve += 1

                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    lr = float(trainer.optimizer.param_groups[0].get("lr", 0.0))

                    # Autotune: detect spikes / plateaus and (optionally) reduce LR.
                    loss_spike = detector.update(loss_val)
                    did_adjust = False
                    reason = ""
                    plateau_event = False
                    if mode in {"monitor", "active"}:
                        if loss_spike:
                            reason = f"spike (loss={loss_val:.6f} > ema+{autotune_spike_std}σ={detector.threshold:.6f})"
                        elif steps_wo_improve == int(autotune_patience):
                            reason = f"plateau (no_improve_steps={steps_wo_improve}, best={best_loss:.6f})"
                            plateau_event = True

                        if reason and mode == "active" and lr > autotune_min_lr:
                            new_lr = _scale_lr(
                                optimizer=trainer.optimizer,
                                scheduler=lr_scheduler,
                                factor=float(autotune_decay),
                                min_lr=float(autotune_min_lr),
                            )
                            detector.reset()
                            detector.update(loss_val)
                            steps_wo_improve = 0
                            did_adjust = True
                            lr = float(new_lr)

                    if reason:
                        # In monitor mode, make it explicit that we'd change LR, but we're not doing it.
                        suffix = ""
                        if mode == "monitor":
                            hypothetical = max(float(autotune_min_lr), float(lr) * float(autotune_decay))
                            suffix = f" → would_set_lr={hypothetical:.3e} (monitor)"
                        msg = (
                            f"[Blockwise Autotune] block={block_index + 1}/{n_blocks} "
                            f"step={step + 1}/{steps_per_block} {reason}"
                        )
                        if did_adjust:
                            logger.warning(msg + f" → lr={lr:.3e}")
                        else:
                            # Only log plateau once; spikes are already discrete events.
                            if not plateau_event and "plateau" in reason:
                                # Should never happen due to == check above, but keep it safe.
                                pass
                            logger.info(msg + suffix)
                    elif autotune_log_every > 0 and (step + 1) % int(autotune_log_every) == 0:
                        logger.info(
                            f"[Blockwise] block={block_index + 1}/{n_blocks} "
                            f"step={step + 1}/{steps_per_block} "
                            f"loss={loss_val:.6f} best={best_loss:.6f} lr={lr:.3e}"
                        )

                    if ctx.inst:
                        scalars: dict[str, float] = {
                            "loss": loss_val,
                            "lr": lr,
                            "block_index": float(block_index),
                            "autotune_spike": float(1.0 if loss_spike else 0.0),
                            # plateau as an event (1.0 only at trigger step)
                            "autotune_plateau": float(1.0 if plateau_event else 0.0),
                            "autotune_lr_adjust": float(1.0 if did_adjust else 0.0),
                        }
                        if convergence_mode:
                            scalars["best_loss"] = float(best_loss)
                            scalars["target_loss"] = float(target_loss)
                        ctx.inst.log_scalars(step=global_step, prefix="train/blockwise", scalars=scalars)

                    if progress is not None and overall_task is not None:
                        progress.update(
                            overall_task,
                            advance=1,
                            description=(
                                f"Block {block_index + 1}/{n_blocks} • "
                                f"step {step + 1}/{steps_per_block} • loss={loss_val:.4f}"
                            ),
                        )
                    if save_every > 0 and global_step % int(save_every) == 0:
                        checkpointer.save(
                            ctx=ctx,
                            run_id=str(run_id),
                            phase="blockwise",
                            step=int(global_step),
                            block_index=block_index,
                            block_step=int(step + 1),
                            global_step=int(global_step),
                            optimizer=trainer.optimizer,
                            scheduler=lr_scheduler,
                        )

                    if convergence_mode:
                        if float(loss_val) <= float(target_loss):
                            logger.success(
                                f"Block {block_index + 1}/{n_blocks} reached target • "
                                f"loss={loss_val:.6f} ≤ {float(target_loss):.6f}"
                            )
                            block_step_end = int(step + 1)
                            break

                        if int(steps_wo_improve) >= int(patience):
                            # Try LR reduction before giving up.
                            if mode == "active" and float(lr) > float(autotune_min_lr):
                                new_lr = _scale_lr(
                                    optimizer=trainer.optimizer,
                                    scheduler=None,
                                    factor=float(autotune_decay),
                                    min_lr=float(autotune_min_lr),
                                )
                                logger.warning(
                                    f"[Blockwise Autotune] plateau before target "
                                    f"(block={block_index + 1}/{n_blocks}, best={best_loss:.6f}, "
                                    f"no_improve_steps={steps_wo_improve}) → lr={float(new_lr):.3e}"
                                )
                                detector.reset()
                                detector.update(loss_val)
                                steps_wo_improve = 0
                                best_loss = loss_val
                                if ctx.inst:
                                    ctx.inst.log_scalars(
                                        step=global_step,
                                        prefix="train/blockwise",
                                        scalars={"autotune_lr_adjust": 1.0},
                                    )
                            else:
                                logger.warning(
                                    f"Block {block_index + 1}/{n_blocks} stopping due to patience "
                                    f"(best={best_loss:.6f}, no_improve_steps={steps_wo_improve})"
                                )
                                block_step_end = int(step + 1)
                                break

                # Ensure we don't carry partial accumulated gradients into the next block.
                trainer.flush_gradients()

                if loss is not None and ctx.dist_ctx is not None:
                    loss = ctx.dist_ctx.all_reduce(loss, op="avg")
                if loss is not None and not convergence_mode:
                    logger.success(f"Block {block_index + 1}/{n_blocks} complete • loss={float(loss):.6f}")

                checkpointer.save(
                    ctx=ctx,
                    run_id=str(run_id),
                    phase="blockwise",
                    step=int(global_step),
                    block_index=block_index,
                    block_step=int(block_step_end),
                    global_step=int(global_step),
                    optimizer=trainer.optimizer,
                    scheduler=lr_scheduler,
                )

        if convergence_mode:
            logger.success(f"Blockwise distillation complete • total_steps={int(global_step)}")
