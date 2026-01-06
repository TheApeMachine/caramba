"""Associative memory dynamics

Runs the settling dynamics for recall and returns the settled state plus
convergence metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.phasor_math import PhasorMath
from synthnn.core.resonant_network import ResonantNetwork


@dataclass(frozen=True, slots=True)
class DynamicsResult:
    """Settling result."""

    final_state: np.ndarray
    steps_run: int
    converged: bool
    mean_phase_delta: float


@dataclass(frozen=True, slots=True)
class MemoryDynamics:
    """Dynamics runner for phase associative memory."""

    num_units: int
    dtype: np.dtype
    projection_eps: float
    coupling_strength: float
    damping: float
    clamp_cue: bool
    clamp_alpha: float | None
    project_each_step: bool
    project_interval: int
    math: PhasorMath

    def settle(
        self,
        *,
        cue_phasor: np.ndarray,
        known: np.ndarray,
        weights: np.ndarray,
        network: ResonantNetwork,
        steps: int,
        dt: float,
        tol: float,
        patience: int,
        use_vectorized: bool,
    ) -> DynamicsResult:
        state = np.zeros(int(self.num_units), dtype=self.dtype)
        state[known] = cue_phasor[known].astype(self.dtype, copy=False)
        state = self.applyClamp(state=state, cue=cue_phasor, known=known)

        prev = self.math.projectUnitOrZero(state, eps=float(self.projection_eps), dtype=self.dtype)
        stable = 0
        last_delta = float("inf")

        for t in range(int(steps)):
            state = self.step(state=state, cue=cue_phasor, known=known, weights=weights, network=network, dt=float(dt), use_vectorized=bool(use_vectorized))
            if bool(self.project_each_step) and (t % int(self.project_interval) == 0):
                state = self.math.projectUnitOrZero(state, eps=float(self.projection_eps), dtype=self.dtype)
            state = self.applyClamp(state=state, cue=cue_phasor, known=known)

            cur = self.math.projectUnitOrZero(state, eps=float(self.projection_eps), dtype=self.dtype)
            last_delta = self.math.meanPhaseDelta(a=cur, b=prev, eps=float(self.projection_eps))
            stable, converged = self.updateConvergence(stable=stable, delta=float(last_delta), tol=float(tol), patience=int(patience))
            prev = cur
            if converged:
                return DynamicsResult(final_state=prev, steps_run=t + 1, converged=True, mean_phase_delta=float(last_delta))

        return DynamicsResult(final_state=prev, steps_run=int(steps), converged=False, mean_phase_delta=float(last_delta if np.isfinite(last_delta) else 0.0))

    def step(
        self,
        *,
        state: np.ndarray,
        cue: np.ndarray,
        known: np.ndarray,
        weights: np.ndarray,
        network: ResonantNetwork,
        dt: float,
        use_vectorized: bool,
    ) -> np.ndarray:
        state = self.applyClamp(state=state, cue=cue, known=known)
        if not bool(use_vectorized):
            return self.stepViaNetwork(state=state, network=network, dt=float(dt))
        coupling = float(self.coupling_strength) * (weights @ state)
        out = state + coupling
        out = out * (1.0 - float(self.damping) * float(dt))
        return out.astype(self.dtype, copy=False)

    def stepViaNetwork(self, *, state: np.ndarray, network: ResonantNetwork, dt: float) -> np.ndarray:
        for i in range(int(self.num_units)):
            network.nodes[self.nodeId(i)].signal = complex(state[i])
        network.step(dt=float(dt))
        out = np.array([network.nodes[self.nodeId(i)].signal for i in range(int(self.num_units))], dtype=self.dtype)
        return out

    def nodeId(self, index: int) -> str:
        return f"mem_{int(index)}"

    def applyClamp(self, *, state: np.ndarray, cue: np.ndarray, known: np.ndarray) -> np.ndarray:
        if not bool(self.clamp_cue):
            return state
        alpha = self.clamp_alpha
        if alpha is None or float(alpha) >= 1.0:
            state[known] = cue[known]
            return state
        state[known] = (1.0 - float(alpha)) * state[known] + float(alpha) * cue[known]
        return state

    def updateConvergence(self, *, stable: int, delta: float, tol: float, patience: int) -> tuple[int, bool]:
        if float(delta) < float(tol):
            stable = int(stable) + 1
            if int(stable) >= int(patience):
                return stable, True
            return stable, False
        return 0, False

