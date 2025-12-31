"""Core orchestrator for training strategy selection.

The Orchestrator is the main control loop that:
1. Monitors training telemetry
2. Decides when to evaluate strategy switches
3. Runs speculative branches to compare strategies
4. Handles rollback when strategies fail
5. Logs decisions for analysis

Two main modes:

1. **Single-run speculative branching**: For single-GPU training
   - Checkpoint at decision boundaries
   - Fork K candidates, run short horizons
   - Pick winner, rollback losers

2. **Population-based** (future): For multi-GPU
   - Run N replicas with different strategies
   - Copy weights/hyperparams from winners to losers
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from caramba.console import logger
from caramba.orchestrator.strategy import (
    DEFAULT_PORTFOLIO,
    Strategy,
    StrategyBundle,
    StrategyState,
    create_strategy,
)
from caramba.orchestrator.telemetry import TelemetrySnapshot, TelemetryStream, TrainingPhase

if TYPE_CHECKING:
    pass


class DecisionBoundary(str, Enum):
    """What triggered a strategy evaluation."""

    PERIODIC = "periodic"  # Regular interval
    SPIKE = "spike"  # Spike detected
    PLATEAU = "plateau"  # Loss plateaued
    PHASE_CHANGE = "phase_change"  # Training phase changed
    SAFETY = "safety"  # Hard safety threshold exceeded (loss/NaN/etc)
    MANUAL = "manual"  # Explicit request


@dataclass
class EvaluationResult:
    """Result of evaluating a strategy on a short horizon."""

    strategy_name: str
    steps: int
    initial_loss: float
    final_loss: float
    loss_improvement: float
    spike_count: int
    avg_grad_norm: float
    score: float  # Overall score for comparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_improvement": self.loss_improvement,
            "spike_count": self.spike_count,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class SwitchReason:
    """Machine-readable explanation for why a switch was considered."""

    trigger_metric: str
    current_value: float | int | str
    threshold: float | int | str
    window: int | None = None
    horizon: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_metric": self.trigger_metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "window": self.window,
            "horizon": self.horizon,
        }


@dataclass
class SwitchDecision:
    """Record of a strategy switch decision."""

    step: int
    from_strategy: str
    to_strategy: str
    reason: DecisionBoundary
    telemetry: dict[str, Any]
    evaluation_results: list[dict[str, Any]]
    switch_reason: SwitchReason | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "from": self.from_strategy,
            "to": self.to_strategy,
            "reason": self.reason.value,
            "switch_reason": self.switch_reason.to_dict() if self.switch_reason else None,
            "telemetry": self.telemetry,
            "evaluations": self.evaluation_results,
            "timestamp": self.timestamp,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Decision boundaries
    decision_interval: int = 500  # Steps between periodic evaluations
    min_steps_between_switches: int = 200  # Hysteresis to prevent thrashing

    # Evaluation
    eval_horizon: int = 100  # Steps to run each candidate
    max_candidates_per_eval: int = 3  # Limit parallel evals

    # Safety
    max_loss_increase: float = 1.5  # Factor above current loss to trigger rollback
    max_spikes_before_switch: int = 3  # Spikes in window to force safety switch
    safety_strategy_name: str = "conservative_adamw"

    # Scoring
    spike_penalty: float = 0.1  # Per-spike penalty in scoring
    compute_cost_penalty: float = 0.0  # Penalty for expensive strategies

    # Logging
    log_dir: Path | None = None
    health_log_interval: int = 100  # Steps between health ticker prints (<=0 disables)

    # Metric subscription keys (metric-agnostic orchestration).
    loss_key: str = "loss"
    grad_norm_key: str = "grad_norm"
    lr_key: str = "lr"


class Orchestrator:
    """Main orchestrator for dynamic strategy selection.

    Usage:

        config = OrchestratorConfig(decision_interval=500)
        orchestrator = Orchestrator(model, config, portfolio)

        for step in range(total_steps):
            # Check if we should evaluate
            if orchestrator.should_evaluate(step, telemetry):
                new_strategy = orchestrator.evaluate_and_switch(
                    current_strategy,
                    dataloader,
                    probe_batch,
                )
                if new_strategy != current_strategy:
                    current_strategy = new_strategy

            # Normal training step
            loss = train_step(current_strategy)
            telemetry = orchestrator.record(loss, grad_norm, lr)
    """

    def __init__(
        self,
        model: nn.Module,
        config: OrchestratorConfig,
        portfolio: list[StrategyBundle] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            model: The model being trained.
            config: Orchestrator configuration.
            portfolio: Strategy bundles to choose from.
        """
        self.model = model
        self.config = config
        self.portfolio = portfolio or DEFAULT_PORTFOLIO

        # Telemetry
        self.telemetry = TelemetryStream(
            total_steps=10000,  # Will be updated
            warmup_steps=0,
        )

        # State tracking
        self._current_strategy_name: str = ""
        self._last_switch_step: int = 0
        self._last_eval_step: int = 0
        self._decision_history: list[SwitchDecision] = []

        # Checkpoint for rollback
        self._checkpoint_state: dict[str, Any] | None = None
        self._checkpoint_strategy_state: StrategyState | None = None

        # Safety tracking
        self._consecutive_failures: int = 0
        self._total_switches: int = 0
        self._loss_baseline: float | None = None

        # Bandit state for strategy selection
        self._strategy_rewards: dict[str, list[float]] = {
            b.name: [] for b in self.portfolio
        }
        self._strategy_pulls: dict[str, int] = {b.name: 0 for b in self.portfolio}

        # Logging
        if config.log_dir is not None:
            config.log_dir.mkdir(parents=True, exist_ok=True)

    def set_total_steps(self, total: int, warmup: int = 0) -> None:
        """Update expected total steps for phase detection."""
        self.telemetry = TelemetryStream(
            total_steps=total,
            warmup_steps=warmup,
        )

    def set_loss_baseline(self, baseline_loss: float | None) -> None:
        """Seed a loss baseline for cold-start safety checks.

        This exists to handle phase boundaries where the first few steps may
        start at an already-bad loss. EMA-based spike detection cannot detect
        that case because the EMA initializes to the first observation.
        """
        if baseline_loss is None:
            self._loss_baseline = None
            return
        v = float(baseline_loss)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"baseline_loss must be finite and > 0, got {baseline_loss!r}")
        self._loss_baseline = v

    def record(
        self,
        *,
        loss: float | None = None,
        grad_norm: float | None = None,
        lr: float | None = None,
        update_norm: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> TelemetrySnapshot:
        """Record training step metrics.

        Returns the current telemetry snapshot.
        """
        if metrics is not None:
            # Populate primary channels from metrics using configured keys.
            if loss is None:
                loss = float(metrics.get(self.config.loss_key, metrics.get("loss", 0.0)))
            if grad_norm is None:
                grad_norm = float(metrics.get(self.config.grad_norm_key, metrics.get("grad_norm", 0.0)))
            if lr is None:
                lr = float(metrics.get(self.config.lr_key, metrics.get("lr", 0.0)))

        if loss is None or grad_norm is None or lr is None:
            raise ValueError("Orchestrator.record requires either (loss, grad_norm, lr) or metrics dict")

        snapshot = self.telemetry.record(
            loss=float(loss),
            grad_norm=float(grad_norm),
            lr=float(lr),
            model=self.model,
            update_norm=update_norm,
            metrics=metrics,
        )
        interval = int(getattr(self.config, "health_log_interval", 0) or 0)
        if interval > 0 and snapshot.step % interval == 0:
            logger.info(
                "[Orchestrator] health "
                f"step={snapshot.step} "
                f"phase={snapshot.phase.value} "
                f"loss_ema={snapshot.loss_ema:.4f} "
                f"loss_slope={snapshot.loss_slope:.6f} "
                f"grad_norm_ema={snapshot.grad_norm_ema:.4f} "
                f"spike_count={snapshot.spike_count}"
            )
        return snapshot

    def should_evaluate(self, step: int, snapshot: TelemetrySnapshot) -> DecisionBoundary | None:
        """Check if we should evaluate strategy switches.

        Returns the decision boundary type, or None if no evaluation needed.
        """
        # Hard safety checks always win, even under hysteresis.
        if not math.isfinite(float(snapshot.loss)):
            return DecisionBoundary.SAFETY
        if self._loss_baseline is not None:
            ceiling = float(self._loss_baseline) * float(self.config.max_loss_increase)
            if float(snapshot.loss) > ceiling:
                return DecisionBoundary.SAFETY

        # Respect hysteresis
        if step - self._last_switch_step < self.config.min_steps_between_switches:
            return None

        # Check for spike-triggered evaluation
        if snapshot.spike_count >= self.config.max_spikes_before_switch:
            return DecisionBoundary.SPIKE

        # Check for plateau
        if (
            snapshot.phase == TrainingPhase.PLATEAU
            and step - self._last_eval_step > self.config.decision_interval // 2
        ):
            return DecisionBoundary.PLATEAU

        # Periodic evaluation
        if step - self._last_eval_step >= self.config.decision_interval:
            return DecisionBoundary.PERIODIC

        return None

    def evaluate_and_switch(
        self,
        current_strategy: Strategy,
        train_step_fn: Any,  # Callable that runs one training step
        eval_fn: Any,  # Callable that evaluates on probe set
        snapshot: TelemetrySnapshot,
        step: int,
        reason: DecisionBoundary,
    ) -> Strategy:
        """Evaluate strategies and potentially switch.

        This runs speculative branches:
        1. Checkpoint current state
        2. Fork K candidate strategies
        3. Run each for eval_horizon steps
        4. Score each on probe set
        5. Return winner (may be current strategy)

        Args:
            current_strategy: The currently active strategy.
            train_step_fn: Function(strategy, step) -> (loss, grad_norm) that runs one step.
            eval_fn: Function(strategy) -> float that evaluates on probe set.
            step: Current global step.
            reason: What triggered this evaluation.

        Returns:
            The winning strategy (may be same as current).
        """
        prev_eval_step = self._last_eval_step
        self._last_eval_step = step

        window = int(getattr(self.telemetry, "window_size", 0) or 0) or None
        horizon = int(getattr(self.config, "eval_horizon", 0) or 0) or None
        if reason == DecisionBoundary.SPIKE:
            reason_detail = SwitchReason(
                trigger_metric="spike_count",
                current_value=int(snapshot.spike_count),
                threshold=int(self.config.max_spikes_before_switch),
                window=window,
                horizon=horizon,
            )
        elif reason == DecisionBoundary.PERIODIC:
            reason_detail = SwitchReason(
                trigger_metric="decision_interval",
                current_value=int(step - prev_eval_step),
                threshold=int(self.config.decision_interval),
                window=window,
                horizon=horizon,
            )
        elif reason == DecisionBoundary.PLATEAU:
            reason_detail = SwitchReason(
                trigger_metric="phase",
                current_value=str(snapshot.phase.value),
                threshold="plateau",
                window=window,
                horizon=horizon,
            )
        elif reason == DecisionBoundary.SAFETY:
            ceiling = (
                float(self._loss_baseline) * float(self.config.max_loss_increase)
                if self._loss_baseline is not None
                else float("nan")
            )
            reason_detail = SwitchReason(
                trigger_metric="loss",
                current_value=float(snapshot.loss),
                threshold=float(ceiling),
                window=window,
                horizon=horizon,
            )
        elif reason == DecisionBoundary.PHASE_CHANGE:
            reason_detail = SwitchReason(
                trigger_metric="phase",
                current_value=str(snapshot.phase.value),
                threshold="phase_change",
                window=window,
                horizon=horizon,
            )
        else:
            reason_detail = SwitchReason(
                trigger_metric=str(reason.value),
                current_value=str(snapshot.phase.value),
                threshold="n/a",
                window=window,
                horizon=horizon,
            )

        logger.info(
            f"[Orchestrator] Evaluating strategies at step {step} "
            f"(reason: {reason.value})"
        )

        # Select candidates to evaluate
        candidates = self._select_candidates(current_strategy.name, snapshot)

        # Always include current strategy
        if current_strategy.name not in [c.name for c in candidates]:
            candidates = [current_strategy.bundle] + candidates[:self.config.max_candidates_per_eval - 1]

        # Checkpoint current state
        self._checkpoint(current_strategy)

        # Evaluate each candidate
        results: list[EvaluationResult] = []

        for bundle in candidates:
            result = self._evaluate_candidate(
                bundle,
                current_strategy,
                train_step_fn,
                eval_fn,
            )
            results.append(result)

            # Restore checkpoint for next candidate
            self._restore(current_strategy)

        # Pick winner
        winner_result = max(results, key=lambda r: r.score)
        winner_bundle = next(b for b in candidates if b.name == winner_result.strategy_name)

        # Log decision
        decision = SwitchDecision(
            step=step,
            from_strategy=current_strategy.name,
            to_strategy=winner_bundle.name,
            reason=reason,
            switch_reason=reason_detail,
            telemetry=snapshot.to_dict(),
            evaluation_results=[r.to_dict() for r in results],
        )
        self._decision_history.append(decision)
        self._log_decision(decision)

        # Update bandit rewards
        for result in results:
            self._update_bandit(result.strategy_name, result.score)

        # Switch if winner is different
        if winner_bundle.name != current_strategy.name:
            logger.log_decision(current_strategy.name, winner_bundle.name, reason_detail)
            logger.success(
                f"[Orchestrator] Switching: {current_strategy.name} → {winner_bundle.name} "
                f"(score: {winner_result.score:.4f})"
            )
            self._last_switch_step = step
            self._total_switches += 1
            return create_strategy(winner_bundle, self.model)

        logger.info(f"[Orchestrator] Keeping current strategy: {current_strategy.name}")
        return current_strategy

    def force_safety_switch(self, current_strategy: Strategy) -> Strategy:
        """Force switch to safety strategy after failures.

        Called when training is unstable and we need to recover.
        """
        logger.warning(
            f"[Orchestrator] Forcing safety switch from {current_strategy.name}"
        )

        safety_bundle = next(
            (b for b in self.portfolio if b.name == self.config.safety_strategy_name),
            self.portfolio[0],
        )

        self._consecutive_failures = 0
        return create_strategy(safety_bundle, self.model)

    def _select_candidates(
        self,
        current_name: str,
        snapshot: TelemetrySnapshot,
    ) -> list[StrategyBundle]:
        """Select candidate strategies to evaluate.

        Uses Thompson Sampling over past rewards.
        """
        # Simple UCB-style selection
        scores: list[tuple[float, StrategyBundle]] = []

        for bundle in self.portfolio:
            pulls = self._strategy_pulls[bundle.name]
            rewards = self._strategy_rewards[bundle.name]

            if pulls == 0:
                # Exploration bonus for untried strategies
                score = float("inf")
            else:
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                # UCB exploration term
                exploration = (2 * torch.log(torch.tensor(sum(self._strategy_pulls.values()) + 1)) / pulls) ** 0.5
                score = avg_reward + float(exploration)

            # Phase-based adjustments
            if snapshot.phase == TrainingPhase.UNSTABLE:
                # Prefer stable strategies
                if "conservative" in bundle.name or "spike_resistant" in bundle.name:
                    score *= 1.5

            if snapshot.phase == TrainingPhase.PLATEAU:
                # Prefer escape strategies
                if "escape" in bundle.name or "aggressive" in bundle.name:
                    score *= 1.3

            scores.append((score, bundle))

        # Sort by score and take top candidates
        scores.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scores[:self.config.max_candidates_per_eval]]

    def _evaluate_candidate(
        self,
        bundle: StrategyBundle,
        current_strategy: Strategy,
        train_step_fn: Any,
        eval_fn: Any,
    ) -> EvaluationResult:
        """Evaluate a candidate strategy on short horizon."""
        # Create strategy
        strategy = create_strategy(bundle, self.model)

        # Copy optimizer state if same family (warm start)
        if bundle.optimizer_family == current_strategy.bundle.optimizer_family:
            try:
                strategy.optimizer.load_state_dict(current_strategy.optimizer.state_dict())
            except Exception:
                pass  # Incompatible states, start fresh

        # Run eval_horizon steps
        losses: list[float] = []
        grad_norms: list[float] = []

        for _ in range(self.config.eval_horizon):
            try:
                loss, grad_norm = train_step_fn(strategy)
                losses.append(loss)
                grad_norms.append(grad_norm)
            except Exception as e:
                logger.warning(f"[Orchestrator] Candidate {bundle.name} failed: {e}")
                # Return bad score for failed strategies
                return EvaluationResult(
                    strategy_name=bundle.name,
                    steps=len(losses),
                    initial_loss=losses[0] if losses else float("inf"),
                    final_loss=float("inf"),
                    loss_improvement=-float("inf"),
                    spike_count=100,
                    avg_grad_norm=float("inf"),
                    score=-float("inf"),
                )

        # Evaluate on probe set
        try:
            probe_loss = eval_fn(strategy)
        except Exception:
            probe_loss = float("inf")

        # Count spikes
        spike_count = sum(
            1 for i in range(1, len(losses))
            if losses[i] > 1.5 * losses[i-1]
        )

        # Compute score
        loss_improvement = losses[0] - losses[-1] if losses else 0.0
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        # Score combines improvement, stability, and probe performance
        score = (
            loss_improvement * 10.0  # Weight improvement
            - spike_count * self.config.spike_penalty  # Penalize spikes
            - probe_loss * 0.1  # Factor in probe loss
        )

        return EvaluationResult(
            strategy_name=bundle.name,
            steps=len(losses),
            initial_loss=losses[0] if losses else 0.0,
            final_loss=losses[-1] if losses else 0.0,
            loss_improvement=loss_improvement,
            spike_count=spike_count,
            avg_grad_norm=avg_grad_norm,
            score=score,
        )

    def _checkpoint(self, strategy: Strategy) -> None:
        """Checkpoint current state for rollback."""
        self._checkpoint_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        self._checkpoint_strategy_state = strategy.capture_state()

    def _restore(self, strategy: Strategy) -> None:
        """Restore from checkpoint."""
        if self._checkpoint_state is None:
            return

        self.model.load_state_dict(self._checkpoint_state["model"])
        torch.set_rng_state(self._checkpoint_state["rng"])
        if self._checkpoint_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(self._checkpoint_state["cuda_rng"])

        if self._checkpoint_strategy_state is not None:
            strategy.restore_state(self._checkpoint_strategy_state)

    def _update_bandit(self, strategy_name: str, reward: float) -> None:
        """Update bandit statistics for a strategy."""
        self._strategy_pulls[strategy_name] += 1
        self._strategy_rewards[strategy_name].append(reward)

        # Keep only recent rewards (non-stationarity)
        max_history = 20
        if len(self._strategy_rewards[strategy_name]) > max_history:
            self._strategy_rewards[strategy_name] = self._strategy_rewards[strategy_name][-max_history:]

    def _log_decision(self, decision: SwitchDecision) -> None:
        """Log a switch decision."""
        if self.config.log_dir is not None:
            log_path = self.config.log_dir / "orchestrator_decisions.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")

    def get_history(self) -> list[SwitchDecision]:
        """Get decision history."""
        return self._decision_history

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_switches": self._total_switches,
            "current_strategy": self._current_strategy_name,
            "strategy_pulls": dict(self._strategy_pulls),
            "strategy_avg_rewards": {
                name: sum(rewards) / len(rewards) if rewards else 0.0
                for name, rewards in self._strategy_rewards.items()
            },
        }
