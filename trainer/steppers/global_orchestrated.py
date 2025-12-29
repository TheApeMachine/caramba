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
            DecisionBoundary,
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
        has_diffusion_teacher = _has_diffusion_head(ctx.teacher)
        use_amp = bool(train.use_amp) and ctx.device.type in ("cuda", "mps", "cpu")
        amp_dtype = autocast_dtype(ctx.device, str(train.amp_dtype))

        scaler = None
        if use_amp and ctx.device.type == "cuda" and amp_dtype == torch.float16:
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                scaler = None

        mode = str(getattr(train, "orchestrator_mode", "active")).lower()
        max_loss_increase = float(getattr(train, "orchestrator_max_loss_increase", 1.5))
        max_spikes = int(getattr(train, "orchestrator_max_spikes_before_switch", 3))
        safety_name = str(getattr(train, "orchestrator_safety_strategy", "spike_resistant"))
        fail_fast = bool(getattr(train, "orchestrator_fail_fast", True))

        orch_config = OrchestratorConfig(
            decision_interval=int(getattr(train, "orchestrator_decision_interval", 500)),
            eval_horizon=int(getattr(train, "orchestrator_eval_horizon", 100)),
            max_loss_increase=float(max_loss_increase),
            max_spikes_before_switch=int(max_spikes),
            safety_strategy_name=str(safety_name),
            log_dir=ctx.checkpoint_dir / "orchestrator",
        )

        # Scale the strategy portfolio around the run-configured base LR.
        # The built-in portfolio bundles are defined around a nominal 1e-4.
        base_lr = float(getattr(train, "orchestrator_portfolio_base_lr", 1e-4))
        run_lr = float(train.lr)
        warmup_steps = int(getattr(train, "warmup_steps", 0))
        total_steps = int(run.steps)

        portfolio: list[StrategyBundle] = []
        for b in DEFAULT_PORTFOLIO:
            lr_mult = float(b.lr) / float(base_lr) if float(base_lr) > 0 else 1.0
            portfolio.append(
                StrategyBundle(
                    **{
                        **b.__dict__,
                        "lr": float(run_lr * lr_mult),
                        "total_steps": total_steps,
                        "warmup_steps": warmup_steps,
                        "use_nowcasting": bool(getattr(train, "orchestrator_use_nowcasting", False)),
                    }
                )
            )

        orchestrator = Orchestrator(model=ctx.student, config=orch_config, portfolio=portfolio)
        orchestrator.set_total_steps(int(run.steps), int(getattr(train, "warmup_steps", 0)))

        initial_name = str(getattr(train, "orchestrator_initial_strategy", "conservative_adamw"))
        initial_bundle = next((b for b in portfolio if b.name == initial_name), portfolio[0])
        current_strategy = create_strategy(initial_bundle, ctx.student)

        if getattr(train, "orchestrator_use_adagc", True):
            current_strategy.add_wrapper(
                AdaGC(ctx.student, warmup_steps=int(getattr(train, "orchestrator_adagc_warmup", 100)))
            )

        logger.header(
            "Global Fine-tuning (Orchestrated)",
            f"{run.steps} steps • mode={mode} • strategy={current_strategy.name}",
        )
        loader_iter = cast(Iterator[TensorDictBase], iter(loader))
        loss: Tensor | None = None

        # Cold-start baseline: compare student against teacher CE on the same batch.
        first_batch, loader_iter = collector.next_batch(loader, loader_iter)
        x0 = first_batch["input_ids"].to(device=ctx.device)
        y0 = first_batch["target_ids"].to(device=ctx.device)
        ctx.teacher.eval()
        ctx.student.eval()
        with torch.no_grad():
            try:
                with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype, enabled=bool(use_amp)):
                    t_loss0, _t_ce0, _t_diff0 = _compute_loss(ctx.teacher, x0, y0, has_diffusion_teacher)
                    s_loss0, _s_ce0, _s_diff0 = _compute_loss(ctx.student, x0, y0, has_diffusion)
            except TypeError:
                t_loss0, _t_ce0, _t_diff0 = _compute_loss(ctx.teacher, x0, y0, has_diffusion_teacher)
                s_loss0, _s_ce0, _s_diff0 = _compute_loss(ctx.student, x0, y0, has_diffusion)

        teacher_loss0 = float(t_loss0.item())
        student_loss0 = float(s_loss0.item())
        orchestrator.set_loss_baseline(teacher_loss0)
        ceiling = teacher_loss0 * float(max_loss_increase)

        if fail_fast and student_loss0 > ceiling:
            raise RuntimeError(
                "Global orchestrator abort: student loss is far above teacher baseline. "
                f"student_loss={student_loss0:.6f}, teacher_loss={teacher_loss0:.6f}, "
                f"ceiling={ceiling:.6f}. "
                "This typically indicates a broken upcycle handoff (covariate shift / logits mismatch). "
                "Fix verification mismatch before global fine-tuning (or relax orchestrator_max_loss_increase / disable orchestrator_fail_fast)."
            )

        ctx.student.train()

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=int(run.steps))
            for step in range(int(run.steps)):
                if step == 0:
                    batch = first_batch
                else:
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
                if reason == DecisionBoundary.SAFETY:
                    if mode == "monitor":
                        raise RuntimeError(
                            "Orchestrator monitor: safety threshold exceeded "
                            f"(loss={float(snapshot.loss):.6f}, baseline={teacher_loss0:.6f}, ceiling={ceiling:.6f})."
                        )
                    if mode == "active":
                        current_strategy = orchestrator.force_safety_switch(current_strategy)
                        if getattr(train, "orchestrator_use_adagc", True):
                            current_strategy.add_wrapper(AdaGC(ctx.student, warmup_steps=50))
                    reason = None  # handled inline; do not run speculative eval this step

                if mode == "active" and reason is not None and val_loader is not None:
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

