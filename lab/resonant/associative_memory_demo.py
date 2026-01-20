"""Associative memory demo (manifest-runnable)

Runs the phase associative memory demo as a Caramba trainer so it can be driven
from a manifest and produce durable artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.manifest import Manifest
from config.target import ExperimentTargetConfig
from console import logger
from resonant.core import PhaseAssociativeMemory


@dataclass
class AssociativeMemoryDemoTrainer:
    """Associative memory demo trainer

    This is an algorithmic demo (no gradient training). It stores K random phase
    patterns then recalls from:
    - noisy cue
    - partial cue (optional rerank)
    """

    output_dir: str = "artifacts/resonant/associative_memory"
    seed: int = 123
    units: int = 64
    patterns: int = 5
    dtype: str = "c128"  # c64|c128
    noise_std: float = 0.65
    known_frac: float = 0.28
    steps: int = 500
    dt: float = 0.05
    rerank_top_k: int = 64

    # PhaseAssociativeMemory hyperparameters (configurable for experiments)
    coupling_strength: float = 0.35
    damping: float = 0.02
    zero_diag: bool = True
    clamp_cue: bool = True
    project_each_step: bool = True
    project_interval: int = 1

    def __post_init__(self) -> None:
        # Normalize numeric fields early so validation is consistent.
        self.units = int(self.units)
        self.patterns = int(self.patterns)
        self.steps = int(self.steps)
        self.project_interval = int(self.project_interval)

        if not (0.0 <= float(self.known_frac) <= 1.0):
            raise ValueError(f"known_frac must be in [0, 1], got {self.known_frac}")
        if float(self.noise_std) < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if int(self.units) <= 0:
            raise ValueError(f"units must be > 0, got {self.units}")
        if int(self.patterns) <= 0:
            raise ValueError(f"patterns must be > 0, got {self.patterns}")
        if int(self.steps) <= 0:
            raise ValueError(f"steps must be > 0, got {self.steps}")
        if float(self.dt) <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: object,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        """Run the associative memory demo (trainer interface).

        Parameters
        ----------
        manifest:
            The experiment `Manifest` driving this run (unused by the demo logic, but
            required by the trainer protocol).
        target:
            Experiment target configuration (unused by the demo logic, but part of the
            trainer protocol).
        engine:
            Execution engine handle (unused; this demo runs locally and produces artifacts).
        dry_run:
            When True, skip execution and return None. This allows orchestrators to
            validate manifests without producing artifacts.

        Returns
        -------
        dict | None
            On success, returns a dict containing artifact/metadata info (e.g. the output
            JSON path). Returns None when `dry_run=True`.
        """
        _ = (manifest, target, engine)
        if dry_run:
            logger.info("Dry run requested, skipping resonant associative memory demo")
            return None

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(int(self.seed))
        n = int(self.units)
        k = int(self.patterns)
        dtype_key = str(self.dtype).strip().lower()
        if dtype_key in {"c64", "complex64"}:
            dtype = np.dtype(np.complex64)
        elif dtype_key in {"c128", "complex128"}:
            dtype = np.dtype(np.complex128)
        else:
            raise ValueError(
                f"Invalid dtype {self.dtype!r}. Expected 'c64' or 'c128' (aliases: 'complex64', 'complex128')."
            )

        angles = rng.uniform(-np.pi, np.pi, size=(k, n)).astype(np.float64)
        pats = np.exp(1j * angles).astype(dtype, copy=False)
        labels = [f"pattern_{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(
            n,
            dtype=dtype,
            label_prefix=None,
            coupling_strength=float(self.coupling_strength),
            damping=float(self.damping),
            zero_diag=bool(self.zero_diag),
            clamp_cue=bool(self.clamp_cue),
            project_each_step=bool(self.project_each_step),
            project_interval=int(self.project_interval),
            node_prefix="mem_",
        )
        mem.store(pats, labels=labels)

        target_idx = int(rng.integers(0, k))
        base = pats[target_idx]

        cue_noisy = np.exp(1j * (np.angle(base) + rng.normal(0.0, float(self.noise_std), size=n))).astype(dtype, copy=False)
        res_noisy = mem.recall(cue_noisy, steps=int(self.steps), dt=float(self.dt), snap=True)

        mask = rng.random(n) < float(self.known_frac)
        cue_partial = base.copy()
        # For unknown elements, use a unit-magnitude complex value with zero phase as a
        # neutral initialization for masked cue entries prior to mem.recall(...).
        cue_partial[~mask] = 1.0 + 0.0j
        res_partial = mem.recall(cue_partial, mask=mask, steps=int(self.steps), dt=float(self.dt), snap=True, rerank_top_k=int(self.rerank_top_k))

        payload = {
            "seed": int(self.seed),
            "units": int(self.units),
            "patterns": int(self.patterns),
            "dtype": str(self.dtype),
            "target_index": int(target_idx),
            "target_label": labels[target_idx],
            "noisy": {"label": res_noisy.label, "index": res_noisy.index, "score": float(res_noisy.score), "steps": int(res_noisy.steps_run), "converged": bool(res_noisy.converged)},
            "partial": {"label": res_partial.label, "index": res_partial.index, "score": float(res_partial.score), "steps": int(res_partial.steps_run), "converged": bool(res_partial.converged), "selection": res_partial.selection},
        }

        out_path = out_dir / "demo_result.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.path(str(out_path), label="resonant_demo")
        return {"artifact": out_path}

