"""Phase associative memory

Public API wrapper that composes storage, dynamics, scoring, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.associative_memory.dynamics import MemoryDynamics
from synthnn.core.associative_memory.scoring import MemoryScorer
from synthnn.core.associative_memory.serialization import MemorySerializer
from synthnn.core.associative_memory.state import StoredMemory
from synthnn.core.associative_memory.storage import MemoryStorage
from synthnn.core.associative_memory.types import RecallResult
from synthnn.core.phasor_math import PhasorMath


@dataclass
class PhaseAssociativeMemory:
    """Phase associative memory

    Stores complex phase patterns as attractors and recalls from noisy or partial
    cues via resonant settling dynamics.
    """

    num_units: int
    dtype: np.dtype = np.dtype(np.complex128)
    label_prefix: str | None = None
    coupling_strength: float = 0.25
    damping: float = 0.02
    zero_diag: bool = True
    clamp_cue: bool = True
    project_each_step: bool = True
    projection_eps: float | None = None
    project_interval: int = 1
    clamp_alpha: float | None = None
    node_prefix: str = "mem_"

    def __post_init__(self) -> None:
        self.num_units = int(self.num_units)
        if self.num_units <= 0:
            raise ValueError("num_units must be > 0")
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("dtype must be complex64 or complex128")
        self.project_interval = max(1, int(self.project_interval))
        self.math = PhasorMath()
        if self.projection_eps is None:
            self.projection_eps = self.math.defaultProjectionEps(dtype=self.dtype)
        if self.clamp_alpha is not None:
            self.clamp_alpha = float(np.clip(float(self.clamp_alpha), 0.0, 1.0))
        self.stored: StoredMemory | None = None

    def store(self, patterns: np.ndarray, labels: list[str] | None = None) -> None:
        """Store patterns and build derived structures."""

        storage = MemoryStorage(
            num_units=int(self.num_units),
            dtype=self.dtype,
            node_prefix=str(self.node_prefix),
            coupling_strength=float(self.coupling_strength),
            damping=float(self.damping),
            zero_diag=bool(self.zero_diag),
            math=self.math,
        )
        self.stored = storage.store(patterns=np.asarray(patterns), labels=labels)

    def recall(
        self,
        cue: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        steps: int = 200,
        dt: float = 0.02,
        snap: bool = True,
        rerank_top_k: int = 0,
        use_vectorized_dynamics: bool = True,
        tol: float = 1e-3,
        patience: int = 8,
    ) -> RecallResult:
        """Recall a stored attractor from a cue."""

        stored = self.requireStored()
        projection_eps = float(self.projection_eps) if self.projection_eps is not None else self.math.defaultProjectionEps(dtype=self.dtype)
        cue_v = self.validateCue(cue=np.asarray(cue))
        known = self.validateMask(mask=mask)
        cue_ph = self.math.toPhasors(cue_v, dtype=self.dtype)

        dyn = MemoryDynamics(
            num_units=int(self.num_units),
            dtype=self.dtype,
            projection_eps=float(projection_eps),
            coupling_strength=float(self.coupling_strength),
            damping=float(self.damping),
            clamp_cue=bool(self.clamp_cue),
            clamp_alpha=self.clamp_alpha,
            project_each_step=bool(self.project_each_step),
            project_interval=int(self.project_interval),
            math=self.math,
        )
        dyn_res = dyn.settle(
            cue_phasor=cue_ph,
            known=known,
            weights=stored.weights,
            network=stored.network,
            steps=int(steps),
            dt=float(dt),
            tol=float(tol),
            patience=int(patience),
            use_vectorized=bool(use_vectorized_dynamics),
        )

        scorer = MemoryScorer(
            num_units=int(self.num_units),
            dtype=self.dtype,
            projection_eps=float(projection_eps),
            label_prefix=self.label_prefix,
        )
        return scorer.finish(
            patterns=stored.patterns,
            labels=stored.labels,
            final_state=dyn_res.final_state,
            known=known,
            snap=bool(snap),
            rerank_top_k=int(rerank_top_k),
            steps_run=int(dyn_res.steps_run),
            converged=bool(dyn_res.converged),
            mean_phase_delta=float(dyn_res.mean_phase_delta),
        )

    def save(self, path: str) -> None:
        """Save memory to an npz artifact."""

        stored = self.requireStored()
        projection_eps = float(self.projection_eps) if self.projection_eps is not None else self.math.defaultProjectionEps(dtype=self.dtype)
        MemorySerializer().save(
            path=str(path),
            num_units=int(self.num_units),
            dtype=self.dtype,
            coupling_strength=float(self.coupling_strength),
            damping=float(self.damping),
            zero_diag=bool(self.zero_diag),
            clamp_cue=bool(self.clamp_cue),
            project_each_step=bool(self.project_each_step),
            projection_eps=float(projection_eps),
            project_interval=int(self.project_interval),
            clamp_alpha=self.clamp_alpha,
            node_prefix=str(self.node_prefix),
            label_prefix=self.label_prefix,
            stored=stored,
        )

    @classmethod
    def load(cls, path: str) -> "PhaseAssociativeMemory":
        """Load memory from an npz artifact."""

        meta = MemorySerializer().load(path=str(path))
        num_units_obj = meta.get("num_units", None)
        if not isinstance(num_units_obj, int):
            raise TypeError("Serialized memory is missing int num_units.")
        num_units = int(num_units_obj)
        patterns = np.asarray(meta["patterns"])
        weights = np.asarray(meta["weights"])
        dtype = np.dtype(patterns.dtype)
        label_prefix_obj = meta.get("label_prefix", None)
        label_prefix = label_prefix_obj if isinstance(label_prefix_obj, str) else None
        clamp_alpha_obj = meta.get("clamp_alpha", None)
        clamp_alpha = float(clamp_alpha_obj) if isinstance(clamp_alpha_obj, float) else None
        mem = cls(
            num_units=num_units,
            dtype=dtype,
            label_prefix=label_prefix,
            coupling_strength=float(meta["coupling_strength"]),  # type: ignore[arg-type]
            damping=float(meta["damping"]),  # type: ignore[arg-type]
            zero_diag=bool(meta["zero_diag"]),  # type: ignore[arg-type]
            clamp_cue=bool(meta["clamp_cue"]),  # type: ignore[arg-type]
            project_each_step=bool(meta["project_each_step"]),  # type: ignore[arg-type]
            projection_eps=float(meta["projection_eps"]),  # type: ignore[arg-type]
            project_interval=int(meta["project_interval"]),  # type: ignore[arg-type]
            clamp_alpha=clamp_alpha,
            node_prefix=str(meta["node_prefix"]),  # type: ignore[arg-type]
        )
        storage = MemoryStorage(
            num_units=int(mem.num_units),
            dtype=mem.dtype,
            node_prefix=str(mem.node_prefix),
            coupling_strength=float(mem.coupling_strength),
            damping=float(mem.damping),
            zero_diag=bool(mem.zero_diag),
            math=mem.math,
        )
        net = storage.networkFromWeights(weights=weights.astype(mem.dtype, copy=False))
        labels = meta.get("labels", None)
        labels_list = labels if isinstance(labels, list) or labels is None else None
        mem.stored = StoredMemory(
            patterns=patterns.astype(mem.dtype, copy=False),
            weights=weights.astype(mem.dtype, copy=False),
            labels=labels_list,  # type: ignore[arg-type]
            network=net,
        )
        return mem

    def requireStored(self) -> StoredMemory:
        if self.stored is None:
            raise RuntimeError("No patterns stored. Call store() first.")
        return self.stored

    def validateCue(self, *, cue: np.ndarray) -> np.ndarray:
        cue = np.asarray(cue)
        if cue.ndim != 1 or cue.shape[0] != int(self.num_units):
            raise ValueError(f"cue must have shape ({self.num_units},)")
        return cue

    def validateMask(self, *, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return np.ones(int(self.num_units), dtype=bool)
        known = np.asarray(mask, dtype=bool)
        if known.shape != (int(self.num_units),):
            raise ValueError(f"mask must have shape ({self.num_units},)")
        return known

