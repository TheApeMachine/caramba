"""SWATS: Switching from Adam to SGD.

Implements the SWATS algorithm from "Improving Generalization Performance by
Switching from Adam to SGD" (https://arxiv.org/abs/1712.07628).

The key insight is that Adam is great for early training (fast convergence)
but SGD with momentum often finds better minima. SWATS automatically detects
when to switch based on the projected gradient norm ratio.

The switch criterion: When the ratio of the projected gradient in the Adam
direction to the full gradient stabilizes, it's time to switch to SGD. The
post-switch SGD learning rate is derived from Adam's state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class SWATSConfig:
    """Configuration for SWATS optimizer."""

    # Adam phase parameters
    adam_lr: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    # SGD phase parameters (lr is derived automatically)
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True

    # Switching parameters
    switch_threshold: float = 1e-9  # Variance threshold for switch
    switch_window: int = 100  # Window for variance computation
    min_steps_before_switch: int = 1000  # Don't switch too early


class SWATS(Optimizer):
    """SWATS optimizer: Adam that switches to SGD.

    This optimizer starts as Adam and automatically switches to SGD when
    the learning process stabilizes. The switch point and post-switch LR
    are determined automatically during training.

    Usage:
        optimizer = SWATS(model.parameters(), config=SWATSConfig())
        for x, y in dataloader:
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if optimizer.has_switched:
                print(f"Switched to SGD at step {optimizer.switch_step}")
    """

    def __init__(
        self,
        params: Any,
        config: SWATSConfig | None = None,
    ) -> None:
        """Initialize SWATS optimizer.

        Args:
            params: Model parameters.
            config: SWATS configuration.
        """
        self.config = config or SWATSConfig()

        defaults = {
            "lr": self.config.adam_lr,
            "betas": self.config.adam_betas,
            "eps": self.config.adam_eps,
            "weight_decay": self.config.weight_decay,
        }
        super().__init__(params, defaults)

        # Switching state
        self._step_count = 0
        self._has_switched = False
        self._switch_step: int | None = None
        self._sgd_lr: float | None = None

        # Ratio tracking for switch detection
        self._ratio_history: list[float] = []

    @property
    def has_switched(self) -> bool:
        """Whether we've switched from Adam to SGD."""
        return self._has_switched

    @property
    def switch_step(self) -> int | None:
        """Step at which we switched (None if not yet)."""
        return self._switch_step

    @property
    def current_mode(self) -> str:
        """Current optimizer mode."""
        return "sgd" if self._has_switched else "adam"

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

        if self._has_switched:
            self._sgd_step()
        else:
            self._adam_step()
            self._check_switch_criterion()

        return loss

    def _adam_step(self) -> None:
        """Perform Adam update and track switch criterion."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["proj_ratio_sum"] = 0.0
                    state["proj_ratio_count"] = 0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute Adam update direction
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1

                # Adam update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Track projection ratio for switch criterion
                # Ratio = <grad, adam_direction> / ||grad||^2
                adam_direction = exp_avg / denom
                grad_norm_sq = grad.norm() ** 2
                if grad_norm_sq > 1e-10:
                    proj = (grad * adam_direction).sum()
                    ratio = float(proj / grad_norm_sq)
                    state["proj_ratio_sum"] += ratio
                    state["proj_ratio_count"] += 1

    def _check_switch_criterion(self) -> None:
        """Check if we should switch from Adam to SGD."""
        if self._step_count < self.config.min_steps_before_switch:
            return

        # Compute average projection ratio across all parameters
        total_ratio = 0.0
        total_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                count = state.get("proj_ratio_count", 0)
                if count > 0:
                    total_ratio += state.get("proj_ratio_sum", 0.0) / count
                    total_count += 1

        if total_count == 0:
            return

        avg_ratio = total_ratio / total_count
        self._ratio_history.append(avg_ratio)

        # Keep only recent history
        if len(self._ratio_history) > self.config.switch_window:
            self._ratio_history = self._ratio_history[-self.config.switch_window:]

        # Check variance of ratio history
        if len(self._ratio_history) >= self.config.switch_window:
            ratios = torch.tensor(self._ratio_history)
            variance = float(ratios.var().item())

            if variance < self.config.switch_threshold:
                # Time to switch!
                self._initiate_switch(avg_ratio)

    def _initiate_switch(self, final_ratio: float) -> None:
        """Switch from Adam to SGD.

        The SGD learning rate is derived from the Adam state following
        the SWATS paper's recommendation.
        """
        self._has_switched = True
        self._switch_step = self._step_count

        # Derive SGD learning rate from Adam's effective step size
        # Use the average ratio to estimate the effective learning rate
        for group in self.param_groups:
            adam_lr = group["lr"]
            # The effective SGD LR is approximately adam_lr / final_ratio
            # but we clip it for safety
            self._sgd_lr = min(adam_lr * 10, max(adam_lr * 0.1, adam_lr / max(0.01, abs(final_ratio))))
            group["lr"] = self._sgd_lr
            break

        # Initialize SGD momentum buffers from Adam's exp_avg
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    # Use Adam's momentum as initial SGD momentum
                    state["momentum_buffer"] = state["exp_avg"].clone()

    def _sgd_step(self) -> None:
        """Perform SGD with momentum update."""
        for group in self.param_groups:
            momentum = self.config.sgd_momentum
            nesterov = self.config.sgd_nesterov
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                p.data.add_(grad, alpha=-lr)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state dict."""
        base_state = super().state_dict()
        base_state["swats_state"] = {
            "step_count": self._step_count,
            "has_switched": self._has_switched,
            "switch_step": self._switch_step,
            "sgd_lr": self._sgd_lr,
            "ratio_history": self._ratio_history,
        }
        return base_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict."""
        swats_state = state_dict.pop("swats_state", {})
        super().load_state_dict(state_dict)

        self._step_count = swats_state.get("step_count", 0)
        self._has_switched = swats_state.get("has_switched", False)
        self._switch_step = swats_state.get("switch_step")
        self._sgd_lr = swats_state.get("sgd_lr")
        self._ratio_history = swats_state.get("ratio_history", [])


class SWATSStrategy:
    """Strategy wrapper for SWATS optimizer.

    Integrates SWATS into the orchestrator's strategy framework.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SWATSConfig | None = None,
    ) -> None:
        """Initialize SWATS strategy.

        Args:
            model: Model to optimize.
            config: SWATS configuration.
        """
        self.model = model
        self.config = config or SWATSConfig()
        self.optimizer = SWATS(model.parameters(), config=self.config)
        self._step_count = 0

    @property
    def name(self) -> str:
        return f"swats_{self.optimizer.current_mode}"

    @property
    def has_switched(self) -> bool:
        return self.optimizer.has_switched

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, loss: Tensor) -> dict[str, float]:
        """Execute one optimization step."""
        self._step_count += 1
        self.optimizer.step()

        # Encode mode as 0.0 for adam, 1.0 for sgd
        mode_float = 1.0 if self.optimizer.current_mode == "sgd" else 0.0

        return {
            "loss": float(loss.item()),
            "lr": self.current_lr,
            "mode": mode_float,
            "has_switched": float(self.optimizer.has_switched),
            "switch_step": float(self.optimizer.switch_step or 0),
        }

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
