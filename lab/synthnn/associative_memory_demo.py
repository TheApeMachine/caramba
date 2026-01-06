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

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.console import logger
from synthnn.core import PhaseAssociativeMemory


@dataclass
class AssociativeMemoryDemoTrainer:
    """Associative memory demo trainer

    This is an algorithmic demo (no gradient training). It stores K random phase
    patterns then recalls from:
    - noisy cue
    - partial cue (optional rerank)
    """

    output_dir: str = "artifacts/synthnn/associative_memory"
    seed: int = 123
    units: int = 64
    patterns: int = 5
    dtype: str = "c128"  # c64|c128
    noise_std: float = 0.65
    known_frac: float = 0.28
    steps: int = 500
    dt: float = 0.05
    rerank_top_k: int = 64

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: object,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        _ = (manifest, target, engine)
        if dry_run:
            logger.info("Dry run requested, skipping SynthNN associative memory demo")
            return None

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(int(self.seed))
        n = int(self.units)
        k = int(self.patterns)
        dtype = np.dtype(np.complex64) if str(self.dtype) == "c64" else np.dtype(np.complex128)

        angles = rng.uniform(-np.pi, np.pi, size=(k, n)).astype(np.float64)
        pats = np.exp(1j * angles).astype(dtype, copy=False)
        labels = [f"pattern_{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(
            n,
            dtype=dtype,
            label_prefix=None,
            coupling_strength=0.35,
            damping=0.02,
            zero_diag=True,
            clamp_cue=True,
            project_each_step=True,
            project_interval=1,
            node_prefix="mem_",
        )
        mem.store(pats, labels=labels)

        target_idx = int(rng.integers(0, k))
        base = pats[target_idx]

        cue_noisy = np.exp(1j * (np.angle(base) + rng.normal(0.0, float(self.noise_std), size=n))).astype(dtype, copy=False)
        res_noisy = mem.recall(cue_noisy, steps=int(self.steps), dt=float(self.dt), snap=True)

        mask = rng.random(n) < float(self.known_frac)
        cue_partial = base.copy()
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
        logger.path(str(out_path), label="synthnn_demo")
        return {"artifact": out_path}

