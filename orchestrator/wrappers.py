"""Gradient wrappers for training stability.

These wrappers modify gradients before the optimizer step to improve
training stability. They can be composed with any optimizer.

Implemented wrappers:

1. **AdaGC** (Adaptive Gradient Clipping): Per-parameter thresholds based on
   EMA of gradient norms. From https://arxiv.org/abs/2502.11034

2. **GradientNoiseInjector**: Adds calibrated noise for escaping local minima.

3. **GradientSmoother**: EMA smoothing of gradients for stability.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from orchestrator.strategy import GradientWrapper


class AdaGC(GradientWrapper):
    """Adaptive Gradient Clipping for spike elimination.

    AdaGC maintains per-parameter EMA of gradient norms and clips gradients
    that exceed a threshold (EMA + k * std). This is more adaptive than
    global norm clipping because different layers can have very different
    gradient scales.

    Reference: https://arxiv.org/abs/2502.11034
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ema_decay: float = 0.99,
        threshold_factor: float = 3.0,
        min_threshold: float = 1e-6,
        warmup_steps: int = 100,
    ) -> None:
        """Initialize AdaGC.

        Args:
            model: The model to clip gradients for.
            ema_decay: Decay factor for gradient norm EMA.
            threshold_factor: Number of std devs above EMA to clip.
            min_threshold: Minimum clipping threshold.
            warmup_steps: Steps before clipping kicks in.
        """
        self.model = model
        self.ema_decay = ema_decay
        self.threshold_factor = threshold_factor
        self.min_threshold = min_threshold
        self.warmup_steps = warmup_steps

        # Per-parameter EMA of gradient norms
        self._grad_norm_ema: dict[str, float] = {}
        self._grad_norm_sq_ema: dict[str, float] = {}  # For variance
        self._step = 0
        self._total_clipped = 0

    def pre_step(
        self,
        model: nn.Module,
        loss: Tensor,
        step: int,
    ) -> dict[str, float]:
        """Clip gradients adaptively before optimizer step.

        Returns metrics about clipping.
        """
        self._step = step
        clipped_count = 0
        total_params = 0
        max_clip_ratio = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            total_params += 1
            grad_norm = float(param.grad.data.norm(2).item())

            # Initialize EMAs
            if name not in self._grad_norm_ema:
                self._grad_norm_ema[name] = grad_norm
                self._grad_norm_sq_ema[name] = grad_norm ** 2
                continue

            # Update EMAs
            self._grad_norm_ema[name] = (
                self.ema_decay * self._grad_norm_ema[name]
                + (1 - self.ema_decay) * grad_norm
            )
            self._grad_norm_sq_ema[name] = (
                self.ema_decay * self._grad_norm_sq_ema[name]
                + (1 - self.ema_decay) * grad_norm ** 2
            )

            # Skip clipping during warmup
            if step < self.warmup_steps:
                continue

            # Compute threshold
            ema = self._grad_norm_ema[name]
            variance = self._grad_norm_sq_ema[name] - ema ** 2
            std = max(0.0, variance) ** 0.5
            threshold = max(
                self.min_threshold,
                ema + self.threshold_factor * std,
            )

            # Clip if necessary
            if grad_norm > threshold:
                clip_ratio = threshold / grad_norm
                param.grad.data.mul_(clip_ratio)
                clipped_count += 1
                max_clip_ratio = max(max_clip_ratio, grad_norm / threshold)
                self._total_clipped += 1

        return {
            "adagc_clipped": float(clipped_count),
            "adagc_clip_ratio": float(clipped_count / max(1, total_params)),
            "adagc_max_clip": max_clip_ratio,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "grad_norm_ema": dict(self._grad_norm_ema),
            "grad_norm_sq_ema": dict(self._grad_norm_sq_ema),
            "step": self._step,
            "total_clipped": self._total_clipped,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serialized state."""
        self._grad_norm_ema = dict(state.get("grad_norm_ema", {}))
        self._grad_norm_sq_ema = dict(state.get("grad_norm_sq_ema", {}))
        self._step = state.get("step", 0)
        self._total_clipped = state.get("total_clipped", 0)


class GradientSmoother(GradientWrapper):
    """EMA smoothing of gradients for stability.

    This is useful when gradients are noisy (small batches, high variance
    data). The smoothed gradient is a weighted average of past gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ema_decay: float = 0.9,
    ) -> None:
        """Initialize gradient smoother.

        Args:
            model: The model.
            ema_decay: EMA decay (higher = more smoothing).
        """
        self.model = model
        self.ema_decay = ema_decay
        self._grad_ema: dict[str, Tensor] = {}
        self._step = 0

    def pre_step(
        self,
        model: nn.Module,
        loss: Tensor,
        step: int,
    ) -> dict[str, float]:
        """Smooth gradients with EMA."""
        self._step = step

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if name not in self._grad_ema:
                self._grad_ema[name] = param.grad.data.clone()
            else:
                self._grad_ema[name] = (
                    self.ema_decay * self._grad_ema[name]
                    + (1 - self.ema_decay) * param.grad.data
                )

            # Replace gradient with smoothed version
            param.grad.data.copy_(self._grad_ema[name])

        return {"grad_smoothed": 1.0}

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "grad_ema": {k: v.cpu() for k, v in self._grad_ema.items()},
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serialized state."""
        grad_ema = state.get("grad_ema", {})
        self._grad_ema = {k: v.to(next(self.model.parameters()).device) for k, v in grad_ema.items()}
        self._step = state.get("step", 0)


class GradientNoiseInjector(GradientWrapper):
    """Inject calibrated noise into gradients.

    This can help escape local minima by adding noise that decays
    over training. The noise scale is proportional to the gradient
    magnitude to avoid destabilizing training.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        initial_scale: float = 0.01,
        decay_rate: float = 0.0001,
        min_scale: float = 0.0,
    ) -> None:
        """Initialize noise injector.

        Args:
            model: The model.
            initial_scale: Initial noise scale (relative to grad norm).
            decay_rate: Exponential decay rate for noise.
            min_scale: Minimum noise scale.
        """
        self.model = model
        self.initial_scale = initial_scale
        self.decay_rate = decay_rate
        self.min_scale = min_scale
        self._step = 0

    def pre_step(
        self,
        model: nn.Module,
        loss: Tensor,
        step: int,
    ) -> dict[str, float]:
        """Inject calibrated noise into gradients."""
        self._step = step

        # Decay noise scale
        scale = max(
            self.min_scale,
            self.initial_scale * (1.0 / (1.0 + self.decay_rate * step)),
        )

        if scale < 1e-8:
            return {"grad_noise_scale": 0.0}

        total_noise = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.data.norm()
            if grad_norm > 0:
                noise = torch.randn_like(param.grad.data) * scale * grad_norm
                param.grad.data.add_(noise)
                total_noise += float(noise.norm().item())

        return {
            "grad_noise_scale": scale,
            "grad_noise_total": total_noise,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = state.get("step", 0)
