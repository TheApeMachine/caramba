from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from config.run import Run
from console import logger
from carmath import global_grad_norm_l2, autocast_dtype
from trainer.collectors import Collector
from trainer.checkpointers import CheckPointer
from trainer.upcycle_context import UpcycleContext
from trainer.steppers.global_stepper import _compute_loss, _has_diffusion_head
from runtime.tensordict_utils import TensorDictBase


class GlobalOrchestratedStepper:
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
        from orchestrator import (
            Orchestrator,
            OrchestratorConfig,
            create_strategy,
            DEFAULT_PORTFOLIO,
            StrategyBundle,
        )
        from orchestrator.wrappers import AdaGC

        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        train = run.train

        loader, val_loader = collector.build_loaders(train, ctx)
        for p in ctx.student.parameters():
            p.requires_grad = True
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

        orch_config = OrchestratorConfig(
            decision_interval=int(getattr(train, "orchestrator_decision_interval", 500)),
            eval_horizon=int(getattr(train, "orchestrator_eval_horizon", 100)),
            log_dir=ctx.checkpoint_dir / "orchestrator",
        )
        orchestrator = Orchestrator(model=ctx.student, config=orch_config, portfolio=DEFAULT_PORTFOLIO)
        orchestrator.set_total_steps(int(run.steps), int(getattr(train, "warmup_steps", 0)))

        initial_name = str(getattr(train, "orchestrator_initial_strategy", "conservative_adamw"))
        initial_bundle = next((b for b in DEFAULT_PORTFOLIO if b.name == initial_name), DEFAULT_PORTFOLIO[0])
        initial_bundle = StrategyBundle(
            **{**initial_bundle.__dict__, "total_steps": int(run.steps), "warmup_steps": int(getattr(train, "warmup_steps", 0)), "lr": float(train.lr)}
        )
        current_strategy = create_strategy(initial_bundle, ctx.student)

        if getattr(train, "orchestrator_use_adagc", True):
            current_strategy.add_wrapper(AdaGC(ctx.student, warmup_steps=100))

        logger.header("Global Fine-tuning (Orchestrated)", f"{run.steps} steps • strategy={current_strategy.name}")
        loader_iter = cast(Iterator[TensorDictBase], iter(loader))
        loss: Tensor | None = None

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(run.steps)):
                batch, loader_iter = collector.next_batch(loader, loader_iter)
                x = batch["input_ids"].to(device=ctx.device)
                y = batch["target_ids"].to(device=ctx.device)

                current_strategy.zero_grad()
                autocast_enabled = bool(use_amp)
                try:
                    with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        loss, ce_loss, diff_loss = _compute_loss(ctx.student, x, y, has_diffusion)
                except TypeError:
                    autocast_enabled = False
                    loss, ce_loss, diff_loss = _compute_loss(ctx.student, x, y, has_diffusion)

                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                grad_norm = global_grad_norm_l2(ctx.student)

                step_metrics = current_strategy.step(loss, scaler=scaler)
                snapshot = orchestrator.record(loss=float(loss.item()), grad_norm=grad_norm, lr=current_strategy.current_lr)

                reason = orchestrator.should_evaluate(step + 1, snapshot)
                if reason is not None and val_loader is not None:
                    def make_train_step():
                        def fn(strategy):
                            nonlocal loader_iter
                            b, loader_iter = collector.next_batch(loader, loader_iter)
                            bx = b["input_ids"].to(device=ctx.device)
                            by = b["target_ids"].to(device=ctx.device)
                            strategy.zero_grad()
                            with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype, enabled=use_amp):
                                logits = ctx.student(bx)
                                batch_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
                            batch_loss.backward()
                            gn = global_grad_norm_l2(ctx.student)
                            strategy.step(batch_loss, scaler=scaler)
                            return float(batch_loss.item()), gn
                        return fn

                    def make_eval_fn():
                        def fn(strategy):
                            ctx.student.eval()
                            total = 0.0
                            count = 0
                            with torch.no_grad():
                                for vb in val_loader:
                                    vx = vb["input_ids"].to(ctx.device)
                                    vy = vb["target_ids"].to(ctx.device)
                                    vlogits = ctx.student(vx)
                                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                                    total += float(vloss.item())
                                    count += 1
                                    if count >= 3:
                                        break
                            ctx.student.train()
                            return total / max(1, count)
                        return fn

                    new_strategy = orchestrator.evaluate_and_switch(
                        current_strategy=current_strategy,
                        train_step_fn=make_train_step(),
                        eval_fn=make_eval_fn(),
                        snapshot=snapshot,
                        step=step + 1,
                        reason=reason,
                    )
                    if new_strategy != current_strategy:
                        current_strategy = new_strategy
                        if getattr(train, "orchestrator_use_adagc", True):
                            current_strategy.add_wrapper(AdaGC(ctx.student, warmup_steps=50))

                loss_val = float(loss.item())
                metrics: dict[str, float] = {
                    "loss": loss_val,
                    "ce_loss": float(ce_loss),
                    "lr": float(current_strategy.current_lr),
                    "grad_norm": grad_norm,
                    "spike_count": float(snapshot.spike_count),
                    **step_metrics,
                }
                if diff_loss is not None:
                    metrics["diff_loss"] = float(diff_loss)
                if ctx.inst:
                    ctx.inst.log_scalars(step=step + 1, prefix="train/global_orch", scalars=metrics)

                progress.update(task, advance=1, description=f"Step {step + 1}/{run.steps} • loss={loss_val:.4f} • strategy={current_strategy.name}")

                if save_every > 0 and (step + 1) % int(save_every) == 0:
                    checkpointer.save(
                        ctx=ctx,
                        run_id=str(run.id),
                        phase="global_orchestrated",
                        step=int(step + 1),
                        global_step=int(step + 1),
                    )

        if loss is not None and ctx.dist_ctx is not None:
            loss = ctx.dist_ctx.all_reduce(loss.detach(), op="avg")  # type: ignore[assignment]
        if loss is not None:
            logger.success(f"Orchestrated training complete • final loss={float(loss):.6f}")

        checkpointer.save(
            ctx=ctx,
            run_id=str(run.id),
            phase="global_orchestrated",
            step=int(run.steps),
            global_step=int(run.steps),
            is_final=True,
        )

