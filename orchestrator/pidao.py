"""PIDAO: PID-Controller Based Adaptive Optimizer.

Implements a control-theoretic optimizer inspired by "Accelerated optimization
in deep learning with a proportional-integral-derivative controller"
(https://www.nature.com/articles/s41467-024-54451-3).

The key insight is that training can be viewed as a dynamical system, and
PID control provides a principled way to navigate the loss landscape:

- P (Proportional): Current gradient direction
- I (Integral): Accumulated gradient history (momentum-like)
- D (Derivative): Change in gradient (acceleration/damping)

This leads to more stable training with interpretable hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class PIDAOConfig:
    """Configuration for PIDAO optimizer."""

    # PID gains
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.01  # Derivative gain

    # Learning rate
    lr: float = 1e-3

    # Regularization
    weight_decay: float = 0.01

    # Stability parameters
    integral_clip: float = 10.0  # Clip integral term to prevent windup
    derivative_smoothing: float = 0.9  # EMA for derivative term

    # Adaptive gains (optional)
    adaptive: bool = False
    adaptive_window: int = 100


class PIDAO(Optimizer):
    """PID-Controller Based Adaptive Optimizer.

    Treats optimization as a control problem where we're trying to drive
    the loss to zero. The PID terms provide:

    - Proportional: React to current error (gradient)
    - Integral: Account for persistent errors (momentum)
    - Derivative: Anticipate future errors (damping)

    Usage:
        optimizer = PIDAO(model.parameters(), config=PIDAOConfig())
        for x, y in dataloader:
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        params: Any,
        config: PIDAOConfig | None = None,
    ) -> None:
        """Initialize PIDAO optimizer.

        Args:
            params: Model parameters.
            config: PIDAO configuration.
        """
        self.config = config or PIDAOConfig()

        defaults = {
            "lr": self.config.lr,
            "kp": self.config.kp,
            "ki": self.config.ki,
            "kd": self.config.kd,
            "weight_decay": self.config.weight_decay,
        }
        super().__init__(params, defaults)

        self._step_count = 0

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        """Perform a single optimization step.

        Args:
            closure: Optional closure for reevaluating the loss.

        Returns:
            Loss value if closure provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            kp = group["kp"]
            ki = group["ki"]
            kd = group["kd"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["integral"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.zeros_like(p.data)
                    state["derivative_ema"] = torch.zeros_like(p.data)

                state["step"] += 1

                integral = state["integral"]
                prev_grad = state["prev_grad"]
                derivative_ema = state["derivative_ema"]

                # Proportional term: current gradient
                p_term = grad

                # Integral term: accumulated gradients (with anti-windup)
                integral.add_(grad)
                # Anti-windup: clip integral to prevent runaway
                integral.clamp_(-self.config.integral_clip, self.config.integral_clip)
                i_term = integral

                # Derivative term: change in gradient (smoothed)
                d_raw = grad - prev_grad
                derivative_ema.mul_(self.config.derivative_smoothing).add_(
                    d_raw, alpha=1 - self.config.derivative_smoothing
                )
                d_term = derivative_ema

                # PID update: u = kp * P + ki * I + kd * D
                update = kp * p_term + ki * i_term + kd * d_term

                # Apply update
                p.data.add_(update, alpha=-lr)

                # Save current gradient for next derivative computation
                state["prev_grad"] = grad.clone()

        return loss

    def reset_integral(self) -> None:
        """Reset integral terms (useful after loss spikes)."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "integral" in state:
                    state["integral"].zero_()

    def get_pid_norms(self) -> dict[str, float]:
        """Get norms of PID terms for diagnostics."""
        p_norm = 0.0
        i_norm = 0.0
        d_norm = 0.0
        count = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state.get(p, {})
                p_norm += float(p.grad.data.norm().item())
                if "integral" in state:
                    i_norm += float(state["integral"].norm().item())
                if "derivative_ema" in state:
                    d_norm += float(state["derivative_ema"].norm().item())
                count += 1

        if count > 0:
            return {
                "p_norm": p_norm / count,
                "i_norm": i_norm / count,
                "d_norm": d_norm / count,
            }
        return {"p_norm": 0.0, "i_norm": 0.0, "d_norm": 0.0}


class AdaptivePIDAO(PIDAO):
    """PIDAO with adaptive gain tuning.

    Automatically adjusts PID gains based on training dynamics:
    - Increase kp when loss is high
    - Increase ki when stuck in plateau
    - Increase kd when oscillating
    """

    def __init__(
        self,
        params: Any,
        config: PIDAOConfig | None = None,
    ) -> None:
        config = config or PIDAOConfig(adaptive=True)
        super().__init__(params, config)

        # Gain adaptation state
        self._loss_history: list[float] = []
        self._gain_update_interval = 50

    def record_loss(self, loss: float) -> None:
        """Record loss for adaptive gain tuning."""
        self._loss_history.append(loss)

        # Keep only recent history
        if len(self._loss_history) > self.config.adaptive_window:
            self._loss_history = self._loss_history[-self.config.adaptive_window:]

        # Periodically update gains
        if self._step_count > 0 and self._step_count % self._gain_update_interval == 0:
            self._adapt_gains()

    def _adapt_gains(self) -> None:
        """Adapt PID gains based on training dynamics."""
        if len(self._loss_history) < 20:
            return

        losses = torch.tensor(self._loss_history)

        # Compute loss statistics
        recent = losses[-20:]
        older = losses[-40:-20] if len(losses) >= 40 else losses[:20]

        recent_mean = float(recent.mean().item())
        recent_std = float(recent.std().item())
        older_mean = float(older.mean().item())

        # Detect regime
        is_improving = recent_mean < older_mean * 0.95
        is_plateau = abs(recent_mean - older_mean) < older_mean * 0.01
        is_oscillating = recent_std > recent_mean * 0.1

        # Adapt gains
        for group in self.param_groups:
            if is_oscillating:
                # More damping, less aggression
                group["kd"] = min(0.1, group["kd"] * 1.1)
                group["kp"] = max(0.5, group["kp"] * 0.95)
            elif is_plateau:
                # More momentum to escape
                group["ki"] = min(0.5, group["ki"] * 1.05)
            elif is_improving:
                # Keep current gains, maybe slightly more aggressive
                group["kp"] = min(2.0, group["kp"] * 1.01)


class PIDAOStrategy:
    """Strategy wrapper for PIDAO optimizer.

    Integrates PIDAO into the orchestrator's strategy framework.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PIDAOConfig | None = None,
        adaptive: bool = False,
    ) -> None:
        """Initialize PIDAO strategy.

        Args:
            model: Model to optimize.
            config: PIDAO configuration.
            adaptive: Use adaptive gain tuning.
        """
        self.model = model
        self.config = config or PIDAOConfig()

        if adaptive:
            self.optimizer = AdaptivePIDAO(model.parameters(), config=self.config)
        else:
            self.optimizer = PIDAO(model.parameters(), config=self.config)

        self._step_count = 0
        self._adaptive = adaptive

    @property
    def name(self) -> str:
        return "pidao_adaptive" if self._adaptive else "pidao"

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, loss: Tensor) -> dict[str, float]:
        """Execute one optimization step."""
        self._step_count += 1
        self.optimizer.step()

        if isinstance(self.optimizer, AdaptivePIDAO):
            self.optimizer.record_loss(float(loss.item()))

        pid_norms = self.optimizer.get_pid_norms()

        return {
            "loss": float(loss.item()),
            "lr": self.current_lr,
            **pid_norms,
        }

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def reset_integral(self) -> None:
        """Reset integral terms (useful after spikes)."""
        self.optimizer.reset_integral()
