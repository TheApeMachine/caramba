"""Universal Memory Tuner.

Implements gradient-free adaptive optimization for memory block parameters
using coordinate descent with momentum-based exploration.
"""

from __future__ import annotations

import math
from typing import Any

from layer.memory_block.memory.telemetry import MemoryHealthTelemetry


class ParameterExplorer:
    """Coordinate descent explorer with integer momentum counter."""

    def __init__(
        self,
        initial_value: float,
        min_value: float,
        max_value: float,
        step_size: float = 0.05,  # Increased from 0.001 to 0.05 (50x bolder)
        patience: int = 2,        # Reduced from 3 for faster switching
        cooldown: int = 10,       # Reduced from 50 to 10 to minimize idle time
        max_momentum: int = 20,   # Increased from 10 to 20 for faster traversal
    ):
        self.value = float(initial_value)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.base_step_size = float(step_size)  # Fixed, never changes
        self.direction = 1.0  # +1 or -1
        self.momentum = 0  # Integer counter: 0 to max_momentum
        self.max_momentum = int(max_momentum)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.steps_without_improvement = 0
        self.steps_since_deactivation = 0
        self.active = True
        self.braking = False  # True when decelerating before reversal

    def step(self, improved: bool) -> float:
        """Take a step based on whether last move improved objective.

        Momentum increments/decrements by 1 each step, acting as multiplier.

        Args:
            improved: True if objective improved, False if degraded

        Returns:
            Delta applied to value
        """
        if not self.active:
            # Check if cooldown period has passed
            self.steps_since_deactivation += 1
            if self.steps_since_deactivation >= self.cooldown:
                # Reactivate
                self.active = True
                self.steps_without_improvement = 0
                self.steps_since_deactivation = 0
                self.momentum = 1  # Start with small momentum
                self.direction = 1.0 if self.value < (self.min_value + self.max_value) / 2 else -1.0
            else:
                return 0.0

        if improved:
            # Accelerate: increase momentum by 1
            self.momentum = min(self.momentum + 1, self.max_momentum)
            self.steps_without_improvement = 0
            self.braking = False  # Not braking if improving
        else:
            # Decelerate: decrease momentum by 1
            self.momentum = max(self.momentum - 1, 0)
            self.steps_without_improvement += 1

            # If momentum reached zero, decide what to do
            if self.momentum == 0:
                if self.braking:
                    # We were braking and have now fully stopped - reverse direction
                    self.direction *= -1.0
                    self.momentum = 1  # Start moving in new direction
                    self.braking = False
                    self.steps_without_improvement = 0
                elif self.steps_without_improvement >= self.patience:
                    # Stop exploring after patience exhausted
                    self.active = False
                    self.steps_since_deactivation = 0
                    return 0.0
                else:
                    # Start braking phase (will reverse after fully stopped)
                    self.braking = True
                    self.momentum = 1  # Continue decelerating

        # Apply step: momentum * base_step_size * direction
        delta = self.direction * self.momentum * self.base_step_size
        new_value = self.value + delta

        # Bounce off boundaries
        if new_value < self.min_value or new_value > self.max_value:
            # Hit boundary - reverse and reduce momentum
            self.direction *= -1.0
            self.momentum = max(1, self.momentum // 2)
            new_value = max(self.min_value, min(self.max_value, new_value))

        actual_delta = new_value - self.value
        self.value = new_value

        return actual_delta

    def reset(self):
        """Reset exploration state."""
        self.steps_without_improvement = 0
        self.momentum = 1
        self.active = True
        self.braking = False


class UniversalMemoryTuner:
    """Cybernetic controller for memory blocks using coordinate descent.

    Uses momentum-based exploration to find optimal parameters.
    """

    def __init__(
        self,
        mode: str = "off",
        ema_decay: float = 0.95,
        warmup_steps: int = 10,
        step_size: float = 0.05,  # Increased from 0.001 to 0.05 (50x bolder)
    ) -> None:
        """Initialize tuner.

        Args:
            mode: "off", "monitor", or "adaptive".
            ema_decay: Smoothing factor for telemetry sensors.
            warmup_steps: Steps to collect baseline before tuning.
            step_size: Fixed step size for parameter exploration.
        """
        self.mode = mode.lower().strip()
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0

        # Sensor EMAs
        self.utilization_ema: float | None = None
        self.conflict_ema: float | None = None
        self.resonant_sim_ema: float | None = None
        self.resonant_steps_ema: float | None = None
        self.vsa_rejection_ema: float | None = None

        # Training metric EMAs
        self.accuracy_ema: float | None = None
        self.loss_ema: float | None = None
        self.loss_variance_ema: float | None = None

        # Objective tracking
        self.objective_ema: float | None = None
        self.prev_objective: float | None = None

        # Parameter explorers (ordered for round-robin)
        self.param_names = ["coupling", "damping", "steps", "novelty", "threshold"]
        self.explorers = {
            # Biased initial values to FORCE activation and signal strength
            "coupling": ParameterExplorer(5.0, 0.1, 20.0, step_size=step_size),  # Start Strong (5x)
            "damping": ParameterExplorer(0.5, 0.1, 50.0, step_size=step_size),   # Start Low (0.5x)
            "steps": ParameterExplorer(0.0, -10.0, 20.0, step_size=1.0),
            "novelty": ParameterExplorer(5.0, 0.1, 10.0, step_size=step_size),   # Start High (5x) to boost eta
            "threshold": ParameterExplorer(0.5, 0.1, 10.0, step_size=step_size), # Start LOW (0.5x) to open gate
        }

        # Coordinate descent: explore one parameter at a time
        self.current_param_index = 0

        # Velocity tracking for visualization (delta per step)
        self.deltas = {k: 0.0 for k in self.explorers.keys()}

    @property
    def resonant_coupling_mult(self) -> float:
        return self.explorers["coupling"].value

    @property
    def resonant_damping_mult(self) -> float:
        return self.explorers["damping"].value

    @property
    def resonant_steps_delta(self) -> int:
        return int(round(self.explorers["steps"].value))

    @property
    def vsa_novelty_mult(self) -> float:
        return self.explorers["novelty"].value

    @property
    def write_threshold_mult(self) -> float:
        return self.explorers["threshold"].value

    def update(self, tel: MemoryHealthTelemetry) -> dict[str, float]:
        """Process new telemetry and update parameter levers."""
        if self.mode == "off":
            return {}

        self.step_count += 1

        # 1. Update Sensor EMAs
        self._update_ema("utilization_ema", tel.utilization)
        self._update_ema("conflict_ema", tel.conflict_rate)

        # Training metrics
        if tel.accuracy is not None:
            self._update_ema("accuracy_ema", tel.accuracy)
        if tel.loss is not None:
            self._update_ema("loss_ema", tel.loss)
        if tel.loss_variance is not None:
            self._update_ema("loss_variance_ema", tel.loss_variance)

        if tel.resonant:
            self._update_ema("resonant_sim_ema", tel.resonant.final_sim)
            self._update_ema("resonant_steps_ema", float(tel.resonant.convergence_steps))

        if tel.vsa:
            self._update_ema("vsa_rejection_ema", tel.vsa.write_rejection_rate)

        # 2. Compute objective function
        objective = self._compute_objective()
        self._update_ema("objective_ema", objective)

        # 3. Adaptive exploration (after warmup)
        if self.mode == "adaptive" and self.step_count > self.warmup_steps:
            improved = self._check_improvement(objective)
            self._explore_parameters(improved)

        self.prev_objective = objective

        # 4. Compile report
        in_warmup = self.step_count <= self.warmup_steps
        report = {
            "tuner/objective": objective,
            "tuner/objective_ema": self.objective_ema or 0.0,
            "tuner/warmup": float(in_warmup),
            "tuner/step_count": float(self.step_count),
            "tuner/resonant_coupling_mult": self.resonant_coupling_mult,
            "tuner/resonant_damping_mult": self.resonant_damping_mult,
            "tuner/resonant_steps_delta": float(self.resonant_steps_delta),
            "tuner/vsa_novelty_mult": self.vsa_novelty_mult,
            "tuner/write_threshold_mult": self.write_threshold_mult,
        }
        return report

    def _update_ema(self, attr: str, value: float) -> None:
        prev = getattr(self, attr)
        if prev is None:
            setattr(self, attr, value)
        else:
            setattr(self, attr, self.ema_decay * prev + (1.0 - self.ema_decay) * value)

    def _compute_objective(self) -> float:
        """Compute scalar objective function to maximize.

        Primary goal: Maximize accuracy and minimize loss variance (stability).
        Secondary: Memory health metrics (Utilization is critical).
        """
        objective = 0.0

        # 1. UTILIZATION (Gatekeeper Metric)
        # If memory is empty, nothing else matters. The tuner must populate it first.
        # Scale: -200 to +20
        if self.utilization_ema is not None:
            util = self.utilization_ema
            if util < 0.2:
                # CRITICAL PENALTY for under-utilization
                # Forces tuner to lower threshold/increase novelty immediately
                objective -= 200.0 * (0.2 - util)  # e.g., if util=0.1, penalty=-20
            elif util > 0.8:
                # Penalty for over-utilization (thrashing)
                objective -= 10.0 * (util - 0.8)
            else:
                # Reward for healthy utilization (0.2 - 0.8)
                objective += 20.0

        # 2. ACCURACY (Performace)
        # Scale: 0-100 points
        if self.accuracy_ema is not None:
            objective += 100.0 * self.accuracy_ema

        # 3. LOSS STABILITY (Reliability)
        # Scale: -50 to 0 points
        if self.loss_variance_ema is not None:
            # Lower variance is better
            objective -= 50.0 * self.loss_variance_ema

        # 4. CONFLICT RATE (Health)
        # Scale: -10 to 0 points
        if self.conflict_ema is not None:
            objective -= 10.0 * self.conflict_ema

        # 5. RESONANT CONVERGENCE (Efficiency)
        # Scale: 0-5 points
        if self.resonant_steps_ema is not None:
            # Reward fast convergence (fewer steps is better)
            # Normalized: 0 steps = 1.0, 20 steps = 0.0
            score = 1.0 - min(1.0, self.resonant_steps_ema / 20.0)
            objective += 5.0 * score

        return objective

    def _check_improvement(self, current_objective: float) -> bool:
        """Check if objective improved since last step."""
        if self.prev_objective is None:
            return True  # First step, assume improvement
        return current_objective > self.prev_objective

    def _explore_parameters(self, improved: bool) -> None:
        """Update one parameter at a time using coordinate descent.

        This prevents oscillations by isolating each parameter's effect.
        """
        # Pick current parameter to explore (round-robin)
        param_name = self.param_names[self.current_param_index]
        explorer = self.explorers[param_name]

        # Step only this parameter
        delta = explorer.step(improved)
        self.deltas[param_name] = delta

        # Move to next parameter for next iteration
        self.current_param_index = (self.current_param_index + 1) % len(self.param_names)

    def get_viz_data(self) -> dict[str, dict[str, float]]:
        """Return structured metrics for console visualization."""
        in_warmup = self.step_count <= self.warmup_steps
        obj_delta = 0.0
        if self.prev_objective is not None and self.objective_ema is not None:
            obj_delta = self.objective_ema - self.prev_objective

        # Count active explorers
        active_count = sum(1 for e in self.explorers.values() if e.active)

        # Get current parameter being explored
        current_param = self.param_names[self.current_param_index]
        current_explorer = self.explorers[current_param]

        return {
            f"Warmup ({self.step_count}/{self.warmup_steps})": {
                "actual": 1.0 if in_warmup else 0.0,
                "target": 0.0,
                "velocity": 0.0
            },
            "Objective": {
                "actual": self.objective_ema or 0.0,
                "target": 0.0,
                "velocity": obj_delta
            },
            f"Active ({active_count}/5)": {
                "actual": float(active_count),
                "target": 0.0,
                "velocity": 0.0
            },
            f"Exploring: {current_param}": {
                "actual": float(current_explorer.momentum),
                "target": 0.0,
                "velocity": 0.0
            },
            "Coupling": {
                "actual": self.resonant_coupling_mult,
                "target": 0.0,
                "velocity": self.deltas["coupling"]
            },
            "Damping": {
                "actual": self.resonant_damping_mult,
                "target": 0.0,
                "velocity": self.deltas["damping"]
            },
            "Novelty": {
                "actual": self.vsa_novelty_mult,
                "target": 0.0,
                "velocity": self.deltas["novelty"]
            },
            "Threshold": {
                "actual": self.write_threshold_mult,
                "target": 0.0,
                "velocity": self.deltas["threshold"]
            },
            "Steps Δ": {
                "actual": float(self.resonant_steps_delta),
                "target": 0.0,
                "velocity": self.deltas["steps"]
            }
        }

    def get_health_metrics(self) -> dict[str, float]:
        """Return health metrics for visualization."""
        return {
            "accuracy": self.accuracy_ema or 0.0,
            "loss_variance": self.loss_variance_ema or 0.0,
            "utilization": self.utilization_ema or 0.0,
            "objective": self.objective_ema or 0.0,
        }


# ============================================================================
# Shared Tuner Singleton
# ============================================================================
# All memory layers should share ONE tuner to prevent competing updates.
# The tuner is updated once per global training step.

_shared_tuner: UniversalMemoryTuner | None = None
_shared_tuner_last_step: int = -1


def get_shared_tuner(mode: str = "adaptive") -> UniversalMemoryTuner:
    """Get or create the shared tuner singleton.

    All memory layers should use this to ensure coordinated optimization.
    """
    global _shared_tuner
    if _shared_tuner is None or _shared_tuner.mode != mode.lower().strip():
        _shared_tuner = UniversalMemoryTuner(mode=mode)
    return _shared_tuner


def should_update_tuner(global_step: int) -> bool:
    """Check if tuner should be updated this step (only once per global step)."""
    global _shared_tuner_last_step
    if global_step != _shared_tuner_last_step:
        _shared_tuner_last_step = global_step
        return True
    return False


def reset_shared_tuner() -> None:
    """Reset the shared tuner (e.g., for new training runs)."""
    global _shared_tuner, _shared_tuner_last_step
    _shared_tuner = None
    _shared_tuner_last_step = -1
