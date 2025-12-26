"""Configuration for the optimizer orchestration layer.

This module defines the configuration schema for dynamic optimizer/scheduler
switching during training.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OrchestratorMode(str, Enum):
    """How the orchestrator operates."""

    DISABLED = "disabled"  # No orchestration
    MONITOR = "monitor"  # Monitor only, no switching
    ACTIVE = "active"  # Full orchestration with switching


class StrategyName(str, Enum):
    """Built-in strategy names."""

    CONSERVATIVE_ADAMW = "conservative_adamw"
    AGGRESSIVE_ADAMW = "aggressive_adamw"
    SGD_ESCAPE = "sgd_escape"
    SPIKE_RESISTANT = "spike_resistant"
    SWATS = "swats"
    PIDAO = "pidao"
    PIDAO_ADAPTIVE = "pidao_adaptive"


class OrchestratorConfig(BaseModel):
    """Configuration for optimizer orchestration.

    The orchestrator dynamically selects and switches between training
    strategies based on telemetry. This enables adaptive training that
    responds to loss spikes, plateaus, and phase changes.

    Example YAML:
        orchestrator:
          mode: active
          decision_interval: 500
          initial_strategy: conservative_adamw
          use_adagc: true
          use_nowcasting: false
    """

    # Mode
    mode: OrchestratorMode = Field(
        default=OrchestratorMode.DISABLED,
        description="Orchestrator operating mode",
    )

    # Decision boundaries
    decision_interval: int = Field(
        default=500,
        ge=10,
        description="Steps between strategy evaluations",
    )
    min_steps_between_switches: int = Field(
        default=200,
        ge=10,
        description="Minimum steps between strategy switches (hysteresis)",
    )

    # Evaluation
    eval_horizon: int = Field(
        default=100,
        ge=10,
        description="Steps to run each candidate during evaluation",
    )
    max_candidates_per_eval: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum strategies to evaluate at each decision point",
    )

    # Initial strategy
    initial_strategy: StrategyName = Field(
        default=StrategyName.CONSERVATIVE_ADAMW,
        description="Starting strategy",
    )

    # Safety
    max_loss_increase: float = Field(
        default=1.5,
        gt=1.0,
        description="Factor above current loss to trigger safety rollback",
    )
    max_spikes_before_switch: int = Field(
        default=3,
        ge=1,
        description="Spike count threshold to force stability switch",
    )
    safety_strategy: StrategyName = Field(
        default=StrategyName.SPIKE_RESISTANT,
        description="Fallback strategy when training is unstable",
    )

    # Stability wrappers
    use_adagc: bool = Field(
        default=True,
        description="Use adaptive gradient clipping (AdaGC)",
    )
    adagc_warmup: int = Field(
        default=100,
        ge=0,
        description="Steps before AdaGC clipping activates",
    )
    adagc_threshold_factor: float = Field(
        default=3.0,
        gt=0,
        description="Std devs above EMA to trigger clipping",
    )

    # Nowcasting (experimental)
    use_nowcasting: bool = Field(
        default=False,
        description="Use weight nowcasting for acceleration",
    )
    nowcast_horizon: int = Field(
        default=50,
        ge=10,
        description="Steps to forecast ahead",
    )
    nowcast_interval: int = Field(
        default=100,
        ge=20,
        description="Steps between nowcast attempts",
    )
    nowcast_max_error: float = Field(
        default=0.1,
        gt=0,
        description="Max forecast error before disabling nowcasting",
    )

    # SWATS-specific
    swats_min_steps: int = Field(
        default=1000,
        ge=100,
        description="Min steps before SWATS can switch Adam→SGD",
    )
    swats_threshold: float = Field(
        default=1e-9,
        gt=0,
        description="Variance threshold for SWATS switching",
    )

    # PIDAO-specific
    pidao_kp: float = Field(
        default=1.0,
        description="PIDAO proportional gain",
    )
    pidao_ki: float = Field(
        default=0.1,
        description="PIDAO integral gain",
    )
    pidao_kd: float = Field(
        default=0.01,
        description="PIDAO derivative gain",
    )

    # Logging
    log_decisions: bool = Field(
        default=True,
        description="Log all strategy switching decisions",
    )

    model_config = {"extra": "forbid"}

    def to_orchestrator_config(self) -> dict[str, Any]:
        """Convert to orchestrator.OrchestratorConfig kwargs."""
        return {
            "decision_interval": self.decision_interval,
            "min_steps_between_switches": self.min_steps_between_switches,
            "eval_horizon": self.eval_horizon,
            "max_candidates_per_eval": self.max_candidates_per_eval,
            "max_loss_increase": self.max_loss_increase,
            "max_spikes_before_switch": self.max_spikes_before_switch,
            "safety_strategy_name": self.safety_strategy.value,
        }


class TelemetryConfig(BaseModel):
    """Configuration for training telemetry.

    Telemetry is collected regardless of orchestrator mode, as it's
    useful for monitoring and analysis.
    """

    enabled: bool = Field(
        default=True,
        description="Enable telemetry collection",
    )
    ema_decay: float = Field(
        default=0.99,
        ge=0.5,
        le=0.9999,
        description="EMA decay for smoothing",
    )
    spike_threshold_std: float = Field(
        default=3.0,
        gt=0,
        description="Std devs above EMA to detect spike",
    )
    window_size: int = Field(
        default=100,
        ge=10,
        description="Window for spike counting and variance",
    )

    model_config = {"extra": "forbid"}
