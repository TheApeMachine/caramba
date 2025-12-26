"""Weight Nowcasting for Training Acceleration.

Implements weight trajectory prediction inspired by:
- WNN (Weight Nowcaster Networks, ICML 2023)
- NiNo (Neuron Interaction and Nowcasting Networks, ICLR 2025)
- Farcasting (2025)

The key insight is that weight trajectories during training are often
predictable. By forecasting future weights, we can "skip" some training
steps, potentially saving significant compute.

This is a simplified version that uses linear extrapolation and
learned correction factors rather than a full neural network predictor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class NowcastConfig:
    """Configuration for weight nowcasting."""

    # Prediction parameters
    horizon: int = 50  # Steps to forecast ahead
    history_size: int = 20  # Steps of history to use

    # When to nowcast
    nowcast_interval: int = 100  # Steps between nowcasts
    min_steps_before_start: int = 200  # Warmup before nowcasting

    # Quality control
    max_forecast_error: float = 0.1  # Max acceptable error before disabling
    error_window: int = 10  # Window for error tracking

    # Correction learning
    learn_correction: bool = True
    correction_lr: float = 0.01


class WeightSnapshot:
    """Snapshot of model weights at a point in time."""

    def __init__(self, model: nn.Module, step: int) -> None:
        """Capture current weights.

        Args:
            model: Model to snapshot.
            step: Current training step.
        """
        self.step = step
        self.weights: dict[str, Tensor] = {}

        for name, param in model.named_parameters():
            self.weights[name] = param.data.clone()

    def diff(self, other: "WeightSnapshot") -> dict[str, Tensor]:
        """Compute weight difference from another snapshot.

        Returns:
            Dict of (self - other) for each parameter.
        """
        diffs = {}
        for name, w in self.weights.items():
            if name in other.weights:
                diffs[name] = w - other.weights[name]
        return diffs

    def norm(self) -> float:
        """Total weight norm."""
        total = 0.0
        for w in self.weights.values():
            total += float(w.norm().item() ** 2)
        return total**0.5


class LinearExtrapolator:
    """Simple linear extrapolation for weight trajectories.

    Fits a linear model to recent weight history and extrapolates
    forward. This is surprisingly effective for smooth training
    trajectories.
    """

    def __init__(self, history_size: int = 20) -> None:
        """Initialize extrapolator.

        Args:
            history_size: Number of snapshots to keep.
        """
        self.history_size = history_size
        self._history: deque[WeightSnapshot] = deque(maxlen=history_size)

    def record(self, snapshot: WeightSnapshot) -> None:
        """Record a weight snapshot."""
        self._history.append(snapshot)

    def can_predict(self) -> bool:
        """Whether we have enough history to predict."""
        return len(self._history) >= 3

    def predict(self, horizon: int) -> dict[str, Tensor] | None:
        """Predict weights `horizon` steps ahead.

        Uses linear regression on weight trajectories.

        Returns:
            Predicted weights, or None if not enough history.
        """
        if not self.can_predict():
            return None

        # Get history as list
        history = list(self._history)
        n = len(history)

        # Compute linear fit for each parameter
        predictions: dict[str, Tensor] = {}
        current_step = history[-1].step

        for name in history[-1].weights.keys():
            # Collect weight trajectory
            steps = torch.tensor([s.step for s in history], dtype=torch.float32)
            weights_stack = torch.stack([s.weights[name].float() for s in history])

            # Normalize steps for numerical stability
            step_mean = steps.mean()
            step_std = steps.std() + 1e-8
            steps_norm = (steps - step_mean) / step_std

            # Linear regression: w = a + b * t
            # Using normal equations
            X = torch.stack([torch.ones(n), steps_norm], dim=1)  # [n, 2]

            # Reshape weights for regression
            orig_shape = weights_stack.shape[1:]
            weights_flat = weights_stack.view(n, -1)  # [n, d]

            # Solve: (X^T X)^{-1} X^T y
            try:
                XtX = X.T @ X
                XtX_inv = torch.linalg.inv(XtX)
                coeffs = XtX_inv @ X.T @ weights_flat  # [2, d]
            except Exception:
                # Fallback to simple extrapolation from last two points
                if n >= 2:
                    velocity = (weights_stack[-1] - weights_stack[-2]).float()
                    predictions[name] = (
                        history[-1].weights[name] + velocity * horizon
                    )
                else:
                    predictions[name] = history[-1].weights[name].clone()
                continue

            # Predict at future step
            future_step = current_step + horizon
            future_step_norm = (future_step - step_mean) / step_std
            X_future = torch.tensor([1.0, future_step_norm])
            pred_flat = X_future @ coeffs  # [d]
            pred = pred_flat.view(orig_shape)

            predictions[name] = pred.to(history[-1].weights[name].dtype)

        return predictions


class WeightNowcaster:
    """Weight nowcasting module for training acceleration.

    Periodically predicts future weights and optionally applies them,
    "skipping" training steps. Includes quality control to disable
    nowcasting when predictions are poor.

    Usage:
        nowcaster = WeightNowcaster(model, config)

        for step in range(total_steps):
            # Normal training step
            train_step()
            nowcaster.record(step)

            # Check if we should nowcast
            if nowcaster.should_nowcast(step):
                skipped = nowcaster.nowcast()
                if skipped > 0:
                    step += skipped  # Adjust step counter
    """

    def __init__(
        self,
        model: nn.Module,
        config: NowcastConfig | None = None,
    ) -> None:
        """Initialize nowcaster.

        Args:
            model: Model to nowcast.
            config: Nowcast configuration.
        """
        self.model = model
        self.config = config or NowcastConfig()

        # Extrapolator
        self._extrapolator = LinearExtrapolator(history_size=self.config.history_size)

        # State
        self._step = 0
        self._last_nowcast_step = 0
        self._enabled = True

        # Quality tracking
        self._prediction_errors: deque[float] = deque(maxlen=self.config.error_window)
        self._last_prediction: dict[str, Tensor] | None = None
        self._last_prediction_step: int | None = None

        # Learned correction (optional)
        self._correction_factors: dict[str, float] = {}

    def record(self, step: int) -> None:
        """Record current weights.

        Call this after each training step.

        Args:
            step: Current training step.
        """
        self._step = step

        # Check prediction accuracy if we made one
        if self._last_prediction is not None and self._last_prediction_step == step:
            self._evaluate_prediction()

        # Record snapshot
        snapshot = WeightSnapshot(self.model, step)
        self._extrapolator.record(snapshot)

    def should_nowcast(self, step: int) -> bool:
        """Check if we should perform nowcasting.

        Args:
            step: Current training step.

        Returns:
            True if nowcasting should be attempted.
        """
        if not self._enabled:
            return False

        if step < self.config.min_steps_before_start:
            return False

        if step - self._last_nowcast_step < self.config.nowcast_interval:
            return False

        if not self._extrapolator.can_predict():
            return False

        return True

    def nowcast(self) -> int:
        """Perform nowcasting: predict and apply future weights.

        Returns:
            Number of steps "skipped" (0 if nowcasting failed).
        """
        if not self._enabled or not self._extrapolator.can_predict():
            return 0

        # Predict future weights
        predictions = self._extrapolator.predict(self.config.horizon)
        if predictions is None:
            return 0

        # Apply corrections if learned
        if self.config.learn_correction:
            for name in predictions:
                if name in self._correction_factors:
                    factor = self._correction_factors[name]
                    # Blend prediction with current weights
                    current = self.model.get_parameter(name.replace(".", "_"))
                    if current is not None:
                        predictions[name] = (
                            factor * predictions[name]
                            + (1 - factor) * current.data
                        )

        # Store prediction for later evaluation
        self._last_prediction = predictions
        self._last_prediction_step = self._step + self.config.horizon

        # Apply predicted weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in predictions:
                    param.data.copy_(predictions[name])

        self._last_nowcast_step = self._step
        return self.config.horizon

    def _evaluate_prediction(self) -> None:
        """Evaluate prediction accuracy and update quality tracking."""
        if self._last_prediction is None:
            return

        # Compute prediction error
        total_error = 0.0
        total_norm = 0.0

        for name, param in self.model.named_parameters():
            if name in self._last_prediction:
                pred = self._last_prediction[name]
                actual = param.data
                error = float((pred - actual).norm().item())
                norm = float(actual.norm().item())
                total_error += error
                total_norm += norm

                # Update correction factor
                if self.config.learn_correction and norm > 1e-10:
                    rel_error = error / norm
                    old_factor = self._correction_factors.get(name, 1.0)
                    # Reduce factor if error is high
                    new_factor = old_factor * (1 - self.config.correction_lr * rel_error)
                    self._correction_factors[name] = max(0.1, min(1.0, new_factor))

        # Track overall error
        if total_norm > 1e-10:
            rel_error = total_error / total_norm
            self._prediction_errors.append(rel_error)

            # Check if we should disable nowcasting
            if len(self._prediction_errors) >= self.config.error_window:
                avg_error = sum(self._prediction_errors) / len(self._prediction_errors)
                if avg_error > self.config.max_forecast_error:
                    self._enabled = False

        # Clear prediction
        self._last_prediction = None
        self._last_prediction_step = None

    def get_stats(self) -> dict[str, Any]:
        """Get nowcasting statistics."""
        errors = list(self._prediction_errors)
        return {
            "enabled": self._enabled,
            "step": self._step,
            "last_nowcast_step": self._last_nowcast_step,
            "avg_error": sum(errors) / len(errors) if errors else 0.0,
            "max_error": max(errors) if errors else 0.0,
            "can_predict": self._extrapolator.can_predict(),
        }

    def enable(self) -> None:
        """Re-enable nowcasting."""
        self._enabled = True
        self._prediction_errors.clear()

    def disable(self) -> None:
        """Disable nowcasting."""
        self._enabled = False
