"""Global fine-tuning loop for the phase-based stepper."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from caramba.carmath import safe_perplexity_from_nll
from caramba.console import logger
from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler
from caramba.trainer.stepper.phase.amp import AmpController
from caramba.trainer.stepper.phase.device import DeviceSynchronizer
from caramba.trainer.stepper.phase.loss import StudentLoss
from caramba.trainer.stepper.phase.optimizer import PhaseOptimizerBuilder
from caramba.trainer.stepper.phase.resume import ResumeManager
from caramba.trainer.stepper.phase.validation import GlobalLossEvaluator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from caramba.config.run import Run
    from caramba.runtime.tensordict_utils import TensorDictBase
    from caramba.trainer.collector.base import Collector
    from caramba.trainer.checkpointer.base import CheckPointer
    from caramba.trainer.upcycle_context import UpcycleContext


class PhaseLoop:
    """Global fine-tuning loop

    Performs standard next-token training (optionally with diffusion auxiliary
    losses) on the full student model during the global fine-tuning phase.
    """

    def __init__(self) -> None:
        """Create a loop with default collaborators."""
        self.amp = None
        self.device_sync = DeviceSynchronizer()
        self.evaluator = GlobalLossEvaluator()
        self.optimizer_builder = PhaseOptimizerBuilder()
        self.resume = ResumeManager()

    def run(
        self,
        *,
        run: "Run",
        ctx: "UpcycleContext",
        collector: "Collector",
        checkpointer: "CheckPointer",
        save_every: int,
        resume_state: dict[str, object] | None,
    ) -> None:
        """Run global fine-tuning for a single run."""
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        train = run.train

        if not hasattr(collector, "build_loaders"):
            raise TypeError("Collector must implement build_loaders(train, ctx).")
        loader, val_loader = collector.build_loaders(train, ctx)  # type: ignore[attr-defined]
        if loader is None:
            raise ValueError("Collector returned no training loader.")

        for p in ctx.student.parameters():
            p.requires_grad = True
        ctx.student.train()

        optimizer = self.optimizer_builder.build(train=train, params=ctx.student.parameters())
        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        self.amp = AmpController(train=train, device=ctx.device)
        loss = StudentLoss(student=ctx.student)

        start_step = self.resume.apply(
            resume_state=resume_state,
            run_id=str(run.id),
            phase="global",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        accum_steps = max(1, int(getattr(train, "gradient_accumulation_steps", 1)))
        best_val_loss = float("inf")
        best_val_ppl = float("inf")

        logger.header("Global Fine-tuning", f"{run.steps} steps")
        loader_iter = iter(loader)
        last_loss: Tensor | None = None

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(start_step), int(run.steps)):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                ce_sum = 0.0
                diff_sum = 0.0
                diff_seen = False

                for _ in range(int(accum_steps)):
                    if not hasattr(collector, "next_batch"):
                        raise TypeError("Collector must implement next_batch(loader, loader_iter).")
                    batch, loader_iter = collector.next_batch(loader, loader_iter)  # type: ignore[attr-defined]
                    x = batch["input_ids"].to(device=ctx.device)
                    y = batch["target_ids"].to(device=ctx.device)

                    with self.amp.context():
                        last_loss, ce_loss, diff_loss = loss.compute(x=x, y=y)

                    loss_sum += float(last_loss.detach())
                    ce_sum += float(ce_loss.detach())
                    if diff_loss is not None:
                        diff_sum += float(diff_loss.detach())
                        diff_seen = True

                    scaled = last_loss / float(accum_steps)
                    self.amp.backward(loss=scaled)

                grad_clip = float(getattr(train, "grad_clip_norm", 0.0))
                if grad_clip > 0.0:
                    self.amp.unscale(optimizer=optimizer)
                    torch.nn.utils.clip_grad_norm_(ctx.student.parameters(), max_norm=float(grad_clip))

                self.amp.step(optimizer=optimizer)
                if scheduler is not None:
                    scheduler.step()
                self.device_sync.sync(device=ctx.device)

                loss_val = float(loss_sum) / float(accum_steps)
                lr = float(optimizer.param_groups[0].get("lr", float(getattr(train, "lr", 0.0))))
                metrics: dict[str, float] = {
                    "loss": float(loss_val),
                    "ce_loss": float(ce_sum) / float(accum_steps),
                    "lr": float(lr),
                }
                if diff_seen:
                    metrics["diff_loss"] = float(diff_sum) / float(accum_steps)
                if ctx.inst is not None:
                    ctx.inst.log_scalars(step=step + 1, prefix="train/global", scalars=metrics)

                eval_every = self.eval_every(ctx=ctx)
                if val_loader is not None and eval_every > 0 and ((step + 1) % int(eval_every) == 0):
                    val_metrics = self.evaluator.evaluate(
                        loss=loss,
                        device=ctx.device,
                        dist_ctx=ctx.dist_ctx,
                        loader=val_loader,
                        max_batches=2,
                    )
                    if ctx.inst is not None:
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

                progress.update(task, advance=1, description=f"Step {step + 1}/{run.steps} • loss={loss_val:.4f}")

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

        if last_loss is not None and ctx.dist_ctx is not None:
            last_loss = ctx.dist_ctx.all_reduce(last_loss.detach(), op="avg")  # type: ignore[assignment]
        if last_loss is not None:
            logger.success(f"Global fine-tuning complete • final loss={float(last_loss):.6f}")

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

    def eval_every(self, *, ctx: "UpcycleContext") -> int:
        """Resolve the validation interval from defaults, if configured."""
        if ctx.defaults is None:
            return 0
        logging = getattr(ctx.defaults, "logging", None)
        if logging is None:
            return 0
        return int(getattr(logging, "eval_iters", 0))
