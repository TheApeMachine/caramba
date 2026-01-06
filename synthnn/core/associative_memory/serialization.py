"""Associative memory serialization

Handles saving/loading PhaseAssociativeMemory state as an npz artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthnn.core.associative_memory.state import StoredMemory


@dataclass(frozen=True, slots=True)
class MemorySerializer:
    """Serialize stored memory + config."""

    def save(
        self,
        *,
        path: str,
        num_units: int,
        dtype: np.dtype,
        coupling_strength: float,
        damping: float,
        zero_diag: bool,
        clamp_cue: bool,
        project_each_step: bool,
        projection_eps: float,
        project_interval: int,
        clamp_alpha: float | None,
        node_prefix: str,
        label_prefix: str | None,
        stored: StoredMemory,
    ) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        labels = [] if stored.labels is None else list(stored.labels)
        np.savez_compressed(
            str(out),
            num_units=np.array([int(num_units)], dtype=np.int64),
            patterns=stored.patterns,
            labels=np.array(labels, dtype=object),
            weights=stored.weights,
            dtype=str(dtype),
            coupling_strength=np.array([float(coupling_strength)], dtype=np.float64),
            damping=np.array([float(damping)], dtype=np.float64),
            zero_diag=np.array([int(zero_diag)], dtype=np.int8),
            clamp_cue=np.array([int(clamp_cue)], dtype=np.int8),
            project_each_step=np.array([int(project_each_step)], dtype=np.int8),
            projection_eps=np.array([float(projection_eps)], dtype=np.float64),
            project_interval=np.array([int(project_interval)], dtype=np.int64),
            clamp_alpha=np.array([(-1.0 if clamp_alpha is None else float(clamp_alpha))], dtype=np.float64),
            node_prefix=np.array([str(node_prefix)], dtype=object),
            label_prefix=np.array(["" if label_prefix is None else str(label_prefix)], dtype=object),
        )

    def load(self, *, path: str) -> dict[str, object]:
        d = np.load(path, allow_pickle=True)
        num_units = int(np.asarray(d["num_units"]).reshape(-1)[0])
        patterns = np.asarray(d["patterns"])
        labels_raw = np.asarray(d["labels"]).tolist() if "labels" in d else []
        labels = None if not labels_raw else [str(x) for x in labels_raw]
        weights = np.asarray(d["weights"])

        clamp_alpha_raw = float(np.asarray(d["clamp_alpha"]).reshape(-1)[0]) if "clamp_alpha" in d else -1.0
        clamp_alpha = None if clamp_alpha_raw < 0 else float(clamp_alpha_raw)
        label_prefix_raw = str(np.asarray(d["label_prefix"]).reshape(-1)[0]) if "label_prefix" in d else ""
        label_prefix = None if not label_prefix_raw else label_prefix_raw

        return {
            "num_units": num_units,
            "patterns": patterns,
            "labels": labels,
            "weights": weights,
            "dtype": patterns.dtype,
            "coupling_strength": float(np.asarray(d["coupling_strength"]).reshape(-1)[0]),
            "damping": float(np.asarray(d["damping"]).reshape(-1)[0]),
            "zero_diag": bool(int(np.asarray(d["zero_diag"]).reshape(-1)[0])),
            "clamp_cue": bool(int(np.asarray(d["clamp_cue"]).reshape(-1)[0])),
            "project_each_step": bool(int(np.asarray(d["project_each_step"]).reshape(-1)[0])),
            "projection_eps": float(np.asarray(d["projection_eps"]).reshape(-1)[0]),
            "project_interval": int(np.asarray(d["project_interval"]).reshape(-1)[0]),
            "clamp_alpha": clamp_alpha,
            "node_prefix": str(np.asarray(d["node_prefix"]).reshape(-1)[0]),
            "label_prefix": label_prefix,
        }

