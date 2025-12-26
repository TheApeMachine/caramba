"""Orchestrated training loop with dynamic strategy selection.

This module integrates the orchestrator into the training loop, enabling
dynamic optimizer/scheduler/clipping strategy switching during training.

The orchestrated trainer wraps the standard training loop and:
1. Monitors telemetry (loss, gradients, spikes)
2. Periodically evaluates strategy switches
3. Handles rollback on failures
4. Logs all decisions for analysis

Usage:

    from trainer.orchestrated import OrchestratedTrainer

    trainer = OrchestratedTrainer(
        model=model,
        config=OrchestratedConfig(
            decision_interval=500,
            eval_horizon=100,
        ),
    )

    for step in range(total_steps):
        x, y = next(dataloader)
        metrics = trainer.step(x, y)
        # Orchestrator automatically handles strategy switching
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from console import logger
from orchestrator import (
    DecisionBoundary,
    Orchestrator,
    OrchestratorConfig,
    TelemetrySnapshot,
)
from orchestrator.strategy import (
    DEFAULT_PORTFOLIO,
    Strategy,
    StrategyBundle,
    create_strategy,
)
from orchestrator.wrappers import AdaGC

if TYPE_CHECKING:
    pass


@dataclass
class OrchestratedConfig:
    """Configuration for orchestrated training."""

    # Orchestrator settings
    decision_interval: int = 500
    eval_horizon: int = 100
    min_steps_between_switches: int = 200
    max_candidates_per_eval: int = 3

    # Initial strategy
    initial_strategy: str = "conservative_adamw"

    # Portfolio (if None, uses default)
    portfolio: list[StrategyBundle] | None = None

    # Training settings
    total_steps: int = 10000
    warmup_steps: int = 0

    # Stability
    use_adagc: bool = True
    adagc_warmup: int = 100

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "auto"

    # Logging
    log_dir: Path | None = None


class OrchestratedTrainer:
    """Training loop with dynamic strategy orchestration.

    This trainer automatically handles:
    - Strategy selection and switching
    - Telemetry monitoring
    - Rollback on failures
    - Gradient clipping/stability
    """

    def __init__(
        self,
        model: nn.Module,
        config: OrchestratedConfig,
        dataloader: DataLoader[tuple[Tensor, Tensor]] | None = None,
        probe_dataloader: DataLoader[tuple[Tensor, Tensor]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize orchestrated trainer.

        Args:
            model: The model to train.
            config: Training configuration.
            dataloader: Training data loader.
            probe_dataloader: Validation data loader for strategy evaluation.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.probe_dataloader = probe_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Portfolio
        self.portfolio = config.portfolio or DEFAULT_PORTFOLIO

        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            model=model,
            config=OrchestratorConfig(
                decision_interval=config.decision_interval,
                eval_horizon=config.eval_horizon,
                min_steps_between_switches=config.min_steps_between_switches,
                max_candidates_per_eval=config.max_candidates_per_eval,
                log_dir=config.log_dir,
            ),
            portfolio=self.portfolio,
        )
        self.orchestrator.set_total_steps(config.total_steps, config.warmup_steps)

        # Initialize current strategy
        initial_bundle = next(
            (b for b in self.portfolio if b.name == config.initial_strategy),
            self.portfolio[0],
        )
        initial_bundle = StrategyBundle(
            **{
                **initial_bundle.__dict__,
                "total_steps": config.total_steps,
                "warmup_steps": config.warmup_steps,
            }
        )
        self.current_strategy = create_strategy(initial_bundle, model)

        # Add AdaGC if configured
        if config.use_adagc:
            adagc = AdaGC(model, warmup_steps=config.adagc_warmup)
            self.current_strategy.add_wrapper(adagc)

        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(config.amp_dtype)
        self.scaler: torch.amp.GradScaler | None = None  # type: ignore[name-defined]
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler()  # type: ignore[attr-defined]

        # State
        self._step = 0
        self._last_snapshot: TelemetrySnapshot | None = None
        self._total_loss = 0.0
        self._data_iter = None

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        """Resolve AMP dtype string."""
        if amp_dtype == "auto":
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        return torch.bfloat16
                except Exception:
                    pass
                return torch.float16
            return torch.bfloat16
        elif amp_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def step(
        self,
        x: Tensor,
        y: Tensor,
        *,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            x: Input batch.
            y: Target batch.
            loss_fn: Optional custom loss function.

        Returns:
            Metrics dict.
        """
        self._step += 1
        self.model.train()

        x = x.to(self.device)
        y = y.to(self.device)

        # Zero gradients
        self.current_strategy.zero_grad()

        # Forward pass with AMP
        with torch.autocast(  # type: ignore[attr-defined]
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(x)
            if loss_fn is not None:
                loss = loss_fn(logits, y)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Compute grad norm before clipping (for telemetry)
        grad_norm = self._compute_grad_norm()

        # Strategy step (includes clipping, optimizer step, scheduler step)
        step_metrics = self.current_strategy.step(loss, scaler=self.scaler)

        # Record telemetry
        self._last_snapshot = self.orchestrator.record(
            loss=float(loss.item()),
            grad_norm=grad_norm,
            lr=self.current_strategy.current_lr,
        )

        # Check for strategy switch
        reason = self.orchestrator.should_evaluate(self._step, self._last_snapshot)
        if reason is not None:
            self._handle_strategy_evaluation(reason)

        # Build metrics
        metrics = {
            "loss": float(loss.item()),
            "grad_norm": grad_norm,
            "lr": self.current_strategy.current_lr,
            "step": float(self._step),
            "strategy": self.current_strategy.name,
            **step_metrics,
        }

        if self._last_snapshot is not None:
            metrics["spike_count"] = float(self._last_snapshot.spike_count)
            metrics["loss_ema"] = self._last_snapshot.loss_ema

        self._total_loss += float(loss.item())
        return metrics

    def _compute_grad_norm(self) -> float:
        """Compute global gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _handle_strategy_evaluation(self, reason: DecisionBoundary) -> None:
        """Handle strategy evaluation at decision boundary."""
        if self.dataloader is None or self.probe_dataloader is None:
            logger.warning(
                "[OrchestratedTrainer] Cannot evaluate strategies without dataloaders"
            )
            return

        # Capture in local vars so closures see non-None types
        dataloader = self.dataloader
        probe_loader = self.probe_dataloader

        def train_step_fn(strategy: Strategy) -> tuple[float, float]:
            """Run one training step with a strategy."""
            if self._data_iter is None:
                self._data_iter = iter(dataloader)

            try:
                x, y = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(dataloader)
                x, y = next(self._data_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            strategy.zero_grad()

            with torch.autocast(  # type: ignore[attr-defined]
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            grad_norm = self._compute_grad_norm()
            strategy.step(loss, scaler=self.scaler)

            return float(loss.item()), grad_norm

        def eval_fn(strategy: Strategy) -> float:
            """Evaluate on probe set."""
            self.model.eval()
            total_loss = 0.0
            count = 0

            with torch.no_grad():
                for x, y in probe_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    total_loss += float(loss.item())
                    count += 1

                    if count >= 5:  # Limit probe batches
                        break

            self.model.train()
            return total_loss / max(1, count)

        # Run evaluation
        new_strategy = self.orchestrator.evaluate_and_switch(
            current_strategy=self.current_strategy,
            train_step_fn=train_step_fn,
            eval_fn=eval_fn,
            step=self._step,
            reason=reason,
        )

        if new_strategy != self.current_strategy:
            self.current_strategy = new_strategy
            # Re-add AdaGC if configured
            if self.config.use_adagc:
                adagc = AdaGC(self.model, warmup_steps=self.config.adagc_warmup)
                self.current_strategy.add_wrapper(adagc)

    def get_telemetry(self) -> TelemetrySnapshot | None:
        """Get the latest telemetry snapshot."""
        return self._last_snapshot

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "step": self._step,
            "avg_loss": self._total_loss / max(1, self._step),
            "current_strategy": self.current_strategy.name,
            "orchestrator": self.orchestrator.get_stats(),
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save trainer state to checkpoint."""
        state = {
            "step": self._step,
            "total_loss": self._total_loss,
            "model_state": self.model.state_dict(),
            "strategy_state": self.current_strategy.capture_state().__dict__,
            "strategy_bundle": self.current_strategy.bundle.to_dict(),
            "orchestrator_stats": self.orchestrator.get_stats(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load trainer state from checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._step = state["step"]
        self._total_loss = state["total_loss"]
        self.model.load_state_dict(state["model_state"])

        # Restore strategy
        bundle_dict = state["strategy_bundle"]
        bundle = StrategyBundle(
            name=bundle_dict.get("name", "default"),
            lr=bundle_dict.get("lr", 1e-4),
            weight_decay=bundle_dict.get("weight_decay", 0.01),
            total_steps=self.config.total_steps,
        )
        self.current_strategy = create_strategy(bundle, self.model)
