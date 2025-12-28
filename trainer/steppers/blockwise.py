from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config.run import Run
from config.train import TrainConfig
from console import logger
from layer.attention import AttentionLayer
from trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
from trainer.collectors import Collector
from trainer.checkpointers import CheckPointer
from trainer.distill import DistillLoss
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from trainer.upcycle_context import UpcycleContext
from carmath import autocast_dtype


def _int_or(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


def _build_blockwise_config(train: TrainConfig, device: torch.device) -> BlockwiseConfig:
    return BlockwiseConfig(
        cache_teacher_outputs=bool(train.cache_teacher_outputs),
        use_amp=bool(train.use_amp),
        amp_dtype=autocast_dtype(device, str(train.amp_dtype)),
        accumulation_steps=int(train.gradient_accumulation_steps),
        # Default-on for speed; can be disabled via TrainConfig.blockwise_truncated_forward if present.
        use_truncated_forward=bool(getattr(train, "blockwise_truncated_forward", True)),
    )


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
        optimizer = torch.optim.AdamW(ctx.student.parameters(), lr=train.lr)

        trainer = BlockwiseTrainer(
            teacher=ctx.teacher,
            student=ctx.student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _, m: isinstance(m, AttentionLayer),
            config=_build_blockwise_config(train, ctx.device),
        )
        n_blocks = trainer.block_count()
        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(n_blocks * run.steps),
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

        if train.convergence_target is not None:
            self._run_convergence(
                run_id=str(run.id),
                trainer=trainer,
                loader=loader,
                loader_iter=loader_iter,
                n_blocks=n_blocks,
                target_loss=float(train.convergence_target),
                patience=int(train.convergence_patience),
                max_steps=int(train.convergence_max_steps),
                ctx=ctx,
                collector=collector,
                checkpointer=checkpointer,
                save_every=int(save_every),
            )
        else:
            self._run_fixed(
                run_id=str(run.id),
                trainer=trainer,
                loader=loader,
                loader_iter=loader_iter,
                n_blocks=n_blocks,
                steps_per_block=int(run.steps),
                lr_scheduler=scheduler,
                ctx=ctx,
                collector=collector,
                checkpointer=checkpointer,
                save_every=int(save_every),
                start_block_index=resume_block_index,
                start_block_step=resume_block_step,
                start_global_step=resume_global_step,
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

    def _run_fixed(
        self,
        *,
        run_id: str,
        trainer: BlockwiseTrainer,
        loader: DataLoader[tuple[Tensor, Tensor]],
        loader_iter: Iterator[tuple[Tensor, Tensor]],
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
    ) -> None:
        total_steps = n_blocks * steps_per_block
        global_step = int(start_global_step)

        with logger.progress_bar() as progress:
            overall_task = progress.add_task(f"Training {n_blocks} blocks...", total=total_steps)
            for block_index in range(int(start_block_index), n_blocks):
                loss = None
                step0 = int(start_block_step) if block_index == int(start_block_index) else 0
                for step in range(step0, steps_per_block):
                    (x, _), loader_iter = collector.next_batch(loader, loader_iter)
                    x = x.to(device=ctx.device)
                    loss = trainer.step(x, block_index=block_index)
                    global_step += 1

                    loss_val = float(loss) if loss is not None else 0.0
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    lr = float(trainer.optimizer.param_groups[0].get("lr", 0.0))
                    if ctx.inst:
                        ctx.inst.log_scalars(
                            step=global_step,
                            prefix="train/blockwise",
                            scalars={"loss": loss_val, "lr": lr, "block_index": float(block_index)},
                        )
                    progress.update(
                        overall_task,
                        advance=1,
                        description=f"Block {block_index + 1}/{n_blocks} • step {step + 1}/{steps_per_block} • loss={loss_val:.4f}",
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

                if loss is not None and ctx.dist_ctx is not None:
                    loss = ctx.dist_ctx.all_reduce(loss, op="avg")
                if loss is not None:
                    logger.success(f"Block {block_index + 1}/{n_blocks} complete • loss={float(loss):.6f}")

                checkpointer.save(
                    ctx=ctx,
                    run_id=str(run_id),
                    phase="blockwise",
                    step=int(global_step),
                    block_index=block_index,
                    block_step=int(steps_per_block),
                    global_step=int(global_step),
                    optimizer=trainer.optimizer,
                    scheduler=lr_scheduler,
                )

    def _run_convergence(
        self,
        *,
        run_id: str,
        trainer: BlockwiseTrainer,
        loader: DataLoader[tuple[Tensor, Tensor]],
        loader_iter: Iterator[tuple[Tensor, Tensor]],
        n_blocks: int,
        target_loss: float,
        patience: int,
        max_steps: int,
        ctx: UpcycleContext,
        collector: Collector,
        checkpointer: CheckPointer,
        save_every: int,
    ) -> None:
        total_steps_taken = 0

        for block_index in range(n_blocks):
            best_loss = float("inf")
            steps_without_improvement = 0
            step = 0

            while step < max_steps:
                (x, _), loader_iter = collector.next_batch(loader, loader_iter)
                x = x.to(device=ctx.device)
                loss = trainer.step(x, block_index=block_index)
                loss_val = float(loss)
                step += 1
                total_steps_taken += 1
                lr = float(trainer.optimizer.param_groups[0].get("lr", 0.0))

                if loss_val < best_loss:
                    best_loss = loss_val
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if ctx.inst:
                    ctx.inst.log_scalars(
                        step=total_steps_taken,
                        prefix="train/blockwise",
                        scalars={
                            "loss": loss_val,
                            "best_loss": best_loss,
                            "target_loss": target_loss,
                            "lr": lr,
                            "block_index": float(block_index),
                        },
                    )

                if save_every > 0 and total_steps_taken % int(save_every) == 0:
                    checkpointer.save(
                        ctx=ctx,
                        run_id=str(run_id),
                        phase="blockwise",
                        step=int(total_steps_taken),
                        block_index=block_index,
                    )

                if loss_val <= target_loss or steps_without_improvement >= patience:
                    break

            checkpointer.save(
                ctx=ctx,
                run_id=str(run_id),
                phase="blockwise",
                step=int(total_steps_taken),
                block_index=block_index,
            )

        logger.success(f"Blockwise distillation complete • total_steps={total_steps_taken}")

