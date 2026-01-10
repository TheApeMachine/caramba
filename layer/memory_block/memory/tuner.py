"""Universal Memory Tuner.

Implements adaptive control loops for memory block parameters based on
telemetry sensors.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from caramba.layer.memory_block.memory.telemetry import MemoryHealthTelemetry


class UniversalMemoryTuner:
    """Cybernetic controller for memory blocks.
    
    Observes MemoryHealthTelemetry and produces parameter scaling factors
    to maintain optimal memory dynamics.
    """

    def __init__(
        self,
        mode: str = "off",
        ema_decay: float = 0.95,
        lever_smoothing: float = 0.99,
        max_delta_per_step: float = 0.01,
        target_utilization: float = 0.5,
        sparsity_budget: float = 0.1,
    ) -> None:
        """Initialize tuner.
        
        Args:
            mode: "off", "monitor", or "adaptive".
            ema_decay: Smoothing factor for telemetry sensors.
            lever_smoothing: Low-pass filter for lever updates (closer to 1.0 is smoother).
            max_delta_per_step: Velocity clamp for lever movements.
            target_utilization: Desired memory bucket usage.
            sparsity_budget: Desired write-rejection rate.
        """
        self.mode = mode.lower().strip()
        self.ema_decay = float(ema_decay)
        self.lever_smoothing = float(lever_smoothing)
        self.max_delta_per_step = float(max_delta_per_step)
        self.target_utilization = float(target_utilization)
        self.sparsity_budget = float(sparsity_budget)
        
        # Sensor EMAs
        self.utilization_ema: float | None = None
        self.conflict_ema: float | None = None
        self.resonant_sim_ema: float | None = None
        self.resonant_steps_ema: float | None = None
        self.vsa_rejection_ema: float | None = None
        
        # Lever Targets (where we WANT to go)
        self.target_resonant_coupling = 1.0
        self.target_resonant_damping = 1.0
        self.target_vsa_novelty = 1.0
        self.target_write_threshold = 1.0
        self.target_resonant_steps_delta = 0.0  # Float for smoothing

        # Actual Levers (where we ARE, used by sub-modules)
        self.resonant_coupling_mult = 1.0
        self.resonant_damping_mult = 1.0
        self.resonant_steps_delta = 0
        self.vsa_novelty_mult = 1.0
        self.write_threshold_mult = 1.0

    def update(self, tel: MemoryHealthTelemetry) -> dict[str, float]:
        """Process new telemetry and update parameter levers."""
        if self.mode == "off":
            return {}

        # 1. Update Sensor EMAs
        self._update_ema("utilization_ema", tel.utilization)
        self._update_ema("conflict_ema", tel.conflict_rate)
        
        if tel.resonant:
            self._update_ema("resonant_sim_ema", tel.resonant.final_sim)
            self._update_ema("resonant_steps_ema", float(tel.resonant.convergence_steps))
            
        if tel.vsa:
            self._update_ema("vsa_rejection_ema", tel.vsa.write_rejection_rate)

        # 2. Update Target Multipliers (Heuristics)
        heuristic_logs: dict[str, float] = {}
        if self.mode == "adaptive":
            heuristic_logs.update(self._run_heuristics())
            
            # 3. Smoothly move Actuals towards Targets (Fluid Dynamics)
            self._apply_smooth_stepping()

        # 4. Compile report
        report = {
            "tuner/resonant_coupling_mult": self.resonant_coupling_mult,
            "tuner/resonant_steps_delta": float(self.resonant_steps_delta),
            "tuner/vsa_novelty_mult": self.vsa_novelty_mult,
            "tuner/write_threshold_mult": self.write_threshold_mult,
            "tuner/target/resonant_coupling": self.target_resonant_coupling,
            "tuner/target/vsa_novelty": self.target_vsa_novelty,
        }
        report.update(heuristic_logs)
        return report

    def _update_ema(self, attr: str, value: float) -> None:
        prev = getattr(self, attr)
        if prev is None:
            setattr(self, attr, value)
        else:
            setattr(self, attr, self.ema_decay * prev + (1.0 - self.ema_decay) * value)

    def _run_heuristics(self) -> dict[str, float]:
        """Update target scaling factors based on EMAs."""
        logs: dict[str, float] = {}

        # --- Resonant Router Logic ---
        if self.resonant_sim_ema is not None:
            if self.resonant_sim_ema < 0.4:
                # Weak coupling -> increase target
                self.target_resonant_coupling = min(5.0, self.target_resonant_coupling * 1.05)
            elif self.resonant_sim_ema > 0.9:
                # Strong overlap -> relax coupling
                self.target_resonant_coupling = max(0.2, self.target_resonant_coupling * 0.98)

        if self.resonant_steps_ema is not None and self.resonant_steps_ema > 15:
            self.target_resonant_steps_delta = min(20.0, self.target_resonant_steps_delta + 0.1)

        # --- Storage/Write Logic ---
        if self.utilization_ema is not None:
            if self.utilization_ema < self.target_utilization * 0.5:
                # Starvation -> lower targets to admit more
                self.target_write_threshold = max(0.1, self.target_write_threshold * 0.95)
                self.target_vsa_novelty = max(0.1, self.target_vsa_novelty * 0.95)
            
            if self.conflict_ema is not None and self.conflict_ema > 0.3:
                # Saturation -> increase targets
                self.target_vsa_novelty = min(10.0, self.target_vsa_novelty * 1.05)
                self.target_write_threshold = min(10.0, self.target_write_threshold * 1.05)

        return logs

    def _apply_smooth_stepping(self) -> None:
        """Low-pass filter and velocity clamp for parameter transitions."""
        levers = [
            ("resonant_coupling_mult", "target_resonant_coupling"),
            ("resonant_damping_mult", "target_resonant_damping"),
            ("vsa_novelty_mult", "target_vsa_novelty"),
            ("write_threshold_mult", "target_write_threshold"),
            ("resonant_steps_delta", "target_resonant_steps_delta"),
        ]

        for actual_name, target_name in levers:
            actual = float(getattr(self, actual_name))
            target = float(getattr(self, target_name))
            
            # 1. Exponential Smoothing (Inertia)
            new_val = self.lever_smoothing * actual + (1.0 - self.lever_smoothing) * target
            
            # 2. Velocity Clamp (Max Delta)
            delta = new_val - actual
            clamped_delta = max(-self.max_delta_per_step, min(self.max_delta_per_step, delta))
            final_val = actual + clamped_delta
            
            # 3. Update state
            # Convert back to int for discrete levers
            if actual_name == "resonant_steps_delta":
                setattr(self, actual_name, int(round(final_val)))
            else:
                setattr(self, actual_name, final_val)
