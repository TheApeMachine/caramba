"""Training strategy abstractions.

A Strategy represents a composable bundle of training components that can be
hot-swapped during training. Each bundle includes:

- Base optimizer (AdamW, SGD, PIDAO, etc.)
- LR controller/scheduler
- Stability wrappers (gradient clipping mode, AdaGC)
- Optional acceleration modules (weight nowcasting)

The key insight is that these components can be independently toggled or
modified, allowing the orchestrator to explore a structured space of
configurations rather than random hyperparameter tuning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler


class OptimizerFamily(str, Enum):
    """Supported optimizer families."""

    ADAMW = "adamw"
    SGD = "sgd"
    SGDM = "sgdm"  # SGD with momentum
    PIDAO = "pidao"  # PID-controller optimizer
    LION = "lion"
    ADAFACTOR = "adafactor"


class ClippingMode(str, Enum):
    """Gradient clipping modes."""

    NONE = "none"
    GLOBAL_NORM = "global_norm"  # Standard global norm clipping
    ADAPTIVE = "adaptive"  # AdaGC-style per-parameter thresholds
    VALUE = "value"  # Per-element value clipping


class SchedulerKind(str, Enum):
    """LR scheduler kinds."""

    NONE = "none"
    COSINE = "cosine"
    LINEAR = "linear"
    WARMUP_COSINE = "warmup_cosine"
    CDAT = "cdat"  # Curvature-dynamics-aware tuner
    HYPERGRADIENT = "hypergradient"  # Online LR tuning via hypergradients


@dataclass
class StrategyState:
    """Serializable state for a strategy, enabling hot-swap and rollback.

    This captures everything needed to restore a strategy to a previous point:
    - Optimizer state dict
    - Scheduler state dict
    - Any wrapper-specific state (e.g., AdaGC EMA of gradients)
    - RNG state for reproducibility
    """

    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    wrapper_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    cumulative_loss: float = 0.0
    best_loss: float = float("inf")

    def capture_rng(self) -> None:
        """Capture current RNG states for reproducibility."""
        self.rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def restore_rng(self) -> None:
        """Restore captured RNG states."""
        if "torch" in self.rng_state:
            torch.set_rng_state(self.rng_state["torch"])
        cuda_state = self.rng_state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass
class StrategyBundle:
    """A complete strategy configuration.

    This is the "genome" of a training strategy—everything needed to
    instantiate and run it.
    """

    # Core optimizer
    optimizer_family: OptimizerFamily = OptimizerFamily.ADAMW
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9  # For SGD/SGDM
    eps: float = 1e-8

    # Scheduler
    scheduler_kind: SchedulerKind = SchedulerKind.COSINE
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    total_steps: int = 10000

    # Clipping
    clipping_mode: ClippingMode = ClippingMode.GLOBAL_NORM
    max_grad_norm: float = 1.0
    adagc_beta: float = 0.99  # EMA decay for AdaGC

    # Toggles
    use_ema_weights: bool = False
    ema_decay: float = 0.9999

    # Acceleration modules
    use_nowcasting: bool = False
    nowcast_horizon: int = 50  # Steps to forecast ahead
    nowcast_interval: int = 100  # How often to nowcast

    # Metadata
    name: str = "default"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/persistence."""
        return {
            "name": self.name,
            "optimizer_family": self.optimizer_family.value,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "momentum": self.momentum,
            "scheduler_kind": self.scheduler_kind.value,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "clipping_mode": self.clipping_mode.value,
            "max_grad_norm": self.max_grad_norm,
            "use_ema_weights": self.use_ema_weights,
            "use_nowcasting": self.use_nowcasting,
        }


@runtime_checkable
class GradientWrapper(Protocol):
    """Protocol for gradient modification wrappers (e.g., AdaGC)."""

    def pre_step(
        self,
        model: nn.Module,
        loss: Tensor,
        step: int,
    ) -> dict[str, float]:
        """Called before optimizer.step(), may modify gradients.

        Returns metrics dict for telemetry.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serialized state."""
        ...


class Strategy(ABC):
    """Abstract base class for training strategies.

    A Strategy manages:
    - Optimizer creation and state
    - Scheduler creation and state
    - Gradient wrappers (clipping, AdaGC)
    - The training step logic

    Strategies are designed to be hot-swappable: the orchestrator can
    capture state, switch to a new strategy, and optionally rollback.
    """

    def __init__(self, bundle: StrategyBundle, model: nn.Module) -> None:
        """Initialize the strategy.

        Args:
            bundle: Configuration for this strategy.
            model: The model being trained.
        """
        self.bundle = bundle
        self.model = model
        self._optimizer: Optimizer | None = None
        self._scheduler: LRScheduler | None = None
        self._wrappers: list[GradientWrapper] = []
        self._step_count = 0
        self._cumulative_loss = 0.0
        self._best_loss = float("inf")

    @property
    def name(self) -> str:
        """Human-readable name for this strategy."""
        return self.bundle.name

    @property
    def optimizer(self) -> Optimizer:
        """Get or create the optimizer."""
        if self._optimizer is None:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    @property
    def scheduler(self) -> "LRScheduler | None":
        """Get or create the scheduler."""
        if self._scheduler is None and self.bundle.scheduler_kind != SchedulerKind.NONE:
            self._scheduler = self._create_scheduler()
        return self._scheduler

    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.param_groups[0].get("lr", self.bundle.lr))

    @abstractmethod
    def _create_optimizer(self) -> Optimizer:
        """Create the optimizer for this strategy."""
        ...

    @abstractmethod
    def _create_scheduler(self) -> "LRScheduler":
        """Create the LR scheduler for this strategy."""
        ...

    def step(
        self,
        loss: Tensor,
        *,
        scaler: torch.amp.GradScaler | None = None,  # type: ignore[name-defined]
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            loss: The loss tensor (already computed).
            scaler: Optional GradScaler for mixed precision.

        Returns:
            Metrics dict with loss, lr, grad_norm, etc.
        """
        self._step_count += 1
        metrics: dict[str, float] = {"loss": float(loss.item())}

        # Run gradient wrappers (may modify gradients)
        for wrapper in self._wrappers:
            wrapper_metrics = wrapper.pre_step(self.model, loss, self._step_count)
            metrics.update(wrapper_metrics)

        # Gradient clipping (if not handled by wrapper)
        # IMPORTANT: Always apply GLOBAL_NORM clipping when requested.
        # Wrappers (e.g. AdaGC) may add additional clipping behavior, but skipping
        # global norm entirely can lead to catastrophic fp16 overflows early in training
        # (especially on MPS where GradScaler is unavailable).
        if self.bundle.clipping_mode == ClippingMode.GLOBAL_NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.bundle.max_grad_norm,
            )
            metrics["grad_norm"] = float(grad_norm)

        # Optimizer step
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        metrics["lr"] = self.current_lr
        metrics["step"] = float(self._step_count)

        # Track cumulative stats
        self._cumulative_loss += float(loss.item())
        if float(loss.item()) < self._best_loss:
            self._best_loss = float(loss.item())

        return metrics

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients before backward pass."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def capture_state(self) -> StrategyState:
        """Capture current state for checkpointing/rollback."""
        state = StrategyState(
            optimizer_state=self.optimizer.state_dict(),
            step=self._step_count,
            cumulative_loss=self._cumulative_loss,
            best_loss=self._best_loss,
        )

        if self.scheduler is not None:
            state.scheduler_state = self.scheduler.state_dict()

        for i, wrapper in enumerate(self._wrappers):
            state.wrapper_states[f"wrapper_{i}"] = wrapper.state_dict()

        state.capture_rng()
        return state

    def restore_state(self, state: StrategyState) -> None:
        """Restore from a captured state."""
        self.optimizer.load_state_dict(state.optimizer_state)
        self._step_count = state.step
        self._cumulative_loss = state.cumulative_loss
        self._best_loss = state.best_loss

        if self.scheduler is not None and state.scheduler_state:
            self.scheduler.load_state_dict(state.scheduler_state)

        for i, wrapper in enumerate(self._wrappers):
            key = f"wrapper_{i}"
            if key in state.wrapper_states:
                wrapper.load_state_dict(state.wrapper_states[key])

        state.restore_rng()

    def add_wrapper(self, wrapper: GradientWrapper) -> None:
        """Add a gradient wrapper (e.g., AdaGC)."""
        self._wrappers.append(wrapper)

    def get_stats(self) -> dict[str, float]:
        """Get strategy statistics."""
        return {
            "step": float(self._step_count),
            "cumulative_loss": self._cumulative_loss,
            "best_loss": self._best_loss,
            "avg_loss": (
                self._cumulative_loss / self._step_count
                if self._step_count > 0
                else 0.0
            ),
            "lr": self.current_lr,
        }


class AdamWStrategy(Strategy):
    """AdamW optimizer strategy."""

    def _create_optimizer(self) -> Optimizer:
        # MPS + fp16 weights + AdamW can poison weights without fp32 master params.
        # Use a master-weight AdamW implementation on MPS when the model has fp16 params.
        try:
            has_fp16 = any(p.dtype == torch.float16 for p in self.model.parameters())
        except Exception:
            has_fp16 = False
        if has_fp16 and next(self.model.parameters()).device.type == "mps":
            from optimizer.adamw_master import AdamWMaster

            return AdamWMaster(
                self.model.parameters(),
                lr=self.bundle.lr,
                betas=self.bundle.betas,
                eps=self.bundle.eps,
                weight_decay=self.bundle.weight_decay,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.bundle.lr,
            betas=self.bundle.betas,
            eps=self.bundle.eps,
            weight_decay=self.bundle.weight_decay,
        )

    def _create_scheduler(self) -> "LRScheduler":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

        match self.bundle.scheduler_kind:
            case SchedulerKind.COSINE | SchedulerKind.WARMUP_COSINE:
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.bundle.total_steps,
                    eta_min=self.bundle.lr * self.bundle.min_lr_ratio,
                )
            case SchedulerKind.LINEAR:
                return LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.bundle.min_lr_ratio,
                    total_iters=self.bundle.total_steps,
                )
            case _:
                # Dummy scheduler that does nothing
                return CosineAnnealingLR(self.optimizer, T_max=1, eta_min=self.bundle.lr)


class SGDStrategy(Strategy):
    """SGD (with optional momentum) strategy."""

    def _create_optimizer(self) -> Optimizer:
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.bundle.lr,
            momentum=self.bundle.momentum,
            weight_decay=self.bundle.weight_decay,
            nesterov=self.bundle.momentum > 0,
        )

    def _create_scheduler(self) -> "LRScheduler":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.bundle.total_steps,
            eta_min=self.bundle.lr * self.bundle.min_lr_ratio,
        )


class LionStrategy(Strategy):
    """Lion optimizer strategy (optionally using fused MPS path)."""

    def _create_optimizer(self) -> Optimizer:
        from optimizer.lion import Lion

        # Use fused path when it can actually run (MPS fp16).
        fused = False
        try:
            p0 = next(self.model.parameters())
            fused = p0.device.type == "mps" and p0.dtype == torch.float16
        except Exception:
            fused = False
        return Lion(
            self.model.parameters(),
            lr=self.bundle.lr,
            betas=self.bundle.betas,
            weight_decay=self.bundle.weight_decay,
            fused=bool(fused),
        )

    def _create_scheduler(self) -> "LRScheduler":
        # Reuse AdamW scheduler choices (LR scheduler is independent of optimizer family).
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

        match self.bundle.scheduler_kind:
            case SchedulerKind.COSINE | SchedulerKind.WARMUP_COSINE:
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.bundle.total_steps,
                    eta_min=self.bundle.lr * self.bundle.min_lr_ratio,
                )
            case SchedulerKind.LINEAR:
                return LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.bundle.min_lr_ratio,
                    total_iters=self.bundle.total_steps,
                )
            case _:
                return CosineAnnealingLR(self.optimizer, T_max=1, eta_min=self.bundle.lr)


def create_strategy(bundle: StrategyBundle, model: nn.Module) -> Strategy:
    """Factory function to create a strategy from a bundle."""
    match bundle.optimizer_family:
        case OptimizerFamily.ADAMW:
            return AdamWStrategy(bundle, model)
        case OptimizerFamily.SGD | OptimizerFamily.SGDM:
            return SGDStrategy(bundle, model)
        case OptimizerFamily.LION:
            return LionStrategy(bundle, model)
        case _:
            # Default to AdamW for unimplemented families
            return AdamWStrategy(bundle, model)


# ============================================================================
# Predefined Strategy Bundles
# ============================================================================


def conservative_adamw() -> StrategyBundle:
    """Conservative AdamW - stable, lower LR, strong clipping."""
    return StrategyBundle(
        name="conservative_adamw",
        description="Stable AdamW with conservative LR and strong clipping",
        optimizer_family=OptimizerFamily.ADAMW,
        lr=1e-5,
        weight_decay=0.01,
        clipping_mode=ClippingMode.GLOBAL_NORM,
        max_grad_norm=0.5,
        scheduler_kind=SchedulerKind.WARMUP_COSINE,
        warmup_steps=100,
    )


def aggressive_adamw() -> StrategyBundle:
    """Aggressive AdamW - higher LR for faster convergence."""
    return StrategyBundle(
        name="aggressive_adamw",
        description="Faster AdamW with higher LR",
        optimizer_family=OptimizerFamily.ADAMW,
        lr=5e-4,
        weight_decay=0.01,
        clipping_mode=ClippingMode.GLOBAL_NORM,
        max_grad_norm=1.0,
        scheduler_kind=SchedulerKind.COSINE,
    )


def sgd_escape() -> StrategyBundle:
    """SGD with momentum - useful for escaping bad basins."""
    return StrategyBundle(
        name="sgd_escape",
        description="SGD with momentum for escaping local minima",
        optimizer_family=OptimizerFamily.SGDM,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        clipping_mode=ClippingMode.GLOBAL_NORM,
        max_grad_norm=1.0,
        scheduler_kind=SchedulerKind.COSINE,
    )


def spike_resistant() -> StrategyBundle:
    """AdamW with AdaGC for spike-prone training."""
    return StrategyBundle(
        name="spike_resistant",
        description="AdamW with adaptive gradient clipping for stability",
        optimizer_family=OptimizerFamily.ADAMW,
        lr=1e-4,
        weight_decay=0.01,
        clipping_mode=ClippingMode.ADAPTIVE,
        adagc_beta=0.99,
        scheduler_kind=SchedulerKind.WARMUP_COSINE,
        warmup_steps=200,
    )


def lion_conservative() -> StrategyBundle:
    """Conservative Lion - stable default for fp16 MPS training."""
    return StrategyBundle(
        name="lion_conservative",
        description="Stable Lion with conservative LR and strong clipping",
        optimizer_family=OptimizerFamily.LION,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        clipping_mode=ClippingMode.GLOBAL_NORM,
        max_grad_norm=0.5,
        scheduler_kind=SchedulerKind.WARMUP_COSINE,
        warmup_steps=200,
    )


DEFAULT_PORTFOLIO: list[StrategyBundle] = [
    conservative_adamw(),
    aggressive_adamw(),
    sgd_escape(),
    spike_resistant(),
    lion_conservative(),
]
