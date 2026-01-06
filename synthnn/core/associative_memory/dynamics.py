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
        # Normalize/validate inputs once; avoid repeated casts in the hot loop.
        steps = int(steps)
        dt = float(dt)
        tol = float(tol)
        patience = int(patience)
        use_vectorized = bool(use_vectorized)
        project_each_step = bool(self.project_each_step)
        project_interval = int(self.project_interval)
        projection_eps = float(self.projection_eps)

        state = np.zeros(int(self.num_units), dtype=self.dtype)
        state[known] = cue_phasor[known].astype(self.dtype, copy=False)
        state = self.applyClamp(state=state, cue=cue_phasor, known=known)

        prev = self.math.projectUnitOrZero(state, eps=projection_eps, dtype=self.dtype)
        stable = 0
        last_delta = float("inf")

        for t in range(steps):
            state = self.step(
                state=state,
                cue=cue_phasor,
                known=known,
                weights=weights,
                network=network,
                dt=dt,
                use_vectorized=use_vectorized,
            )
            # Avoid modulo-by-zero when project_interval is 0 (or negative).
            if project_each_step and project_interval > 0 and (t % project_interval == 0):
                state = self.math.projectUnitOrZero(state, eps=projection_eps, dtype=self.dtype)
            state = self.applyClamp(state=state, cue=cue_phasor, known=known)

            cur = self.math.projectUnitOrZero(state, eps=projection_eps, dtype=self.dtype)
            last_delta = self.math.meanPhaseDelta(a=cur, b=prev, eps=projection_eps)
            stable, converged = self.updateConvergence(
                stable=stable, delta=float(last_delta), tol=tol, patience=patience
            )
            prev = cur
            if converged:
                return DynamicsResult(final_state=prev, steps_run=t + 1, converged=True, mean_phase_delta=float(last_delta))

        return DynamicsResult(
            final_state=prev,
            steps_run=steps,
            converged=False,
            mean_phase_delta=float(last_delta if np.isfinite(last_delta) else 0.0),
        )

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
        if not use_vectorized:
            return self.stepViaNetwork(state=state, network=network, dt=float(dt))
        coupling = float(self.coupling_strength) * (weights @ state)
        out = state + coupling
        out = out * (1.0 - float(self.damping) * float(dt))
        return out.astype(self.dtype, copy=False)

    def stepViaNetwork(self, *, state: np.ndarray, network: ResonantNetwork, dt: float) -> np.ndarray:
        num_units = int(self.num_units)
        # Validate node presence for clearer failures than KeyError.
        for i in range(num_units):
            node_key = self.nodeId(i)
            if node_key not in network.nodes:
                raise KeyError(
                    f"Missing node {node_key!r} in network {type(network).__name__}({getattr(network, 'name', '<unnamed>')!r}) at index i={i}"
                )
            network.nodes[node_key].signal = complex(state[i])
        network.step(dt=float(dt))
        out: list[complex] = []
        for i in range(num_units):
            node_key = self.nodeId(i)
            if node_key not in network.nodes:
                raise KeyError(
                    f"Missing node {node_key!r} in network {type(network).__name__}({getattr(network, 'name', '<unnamed>')!r}) at index i={i}"
                )
            out.append(network.nodes[node_key].signal)
        return np.asarray(out, dtype=self.dtype)

    def nodeId(self, index: int) -> str:
        return f"mem_{int(index)}"

    def applyClamp(self, *, state: np.ndarray, cue: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Optionally clamp known cue elements into the state.

        This method returns a (possibly) modified array. When clamping is enabled,
        it **does not mutate** the input `state`; it operates on a copy to avoid
        surprising callers.
        """
        if not bool(self.clamp_cue):
            return state
        state = np.copy(state)
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

