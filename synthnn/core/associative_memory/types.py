"""Associative memory types

Defines result payloads produced by recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Recall result

    Collects recall selection metadata and states so callers can analyze
    convergence and selection behavior.
    """

    label: str | None
    index: int | None
    score: float
    scores: np.ndarray
    masked_scores: np.ndarray | None
    selection: str
    final_state: np.ndarray
    snapped_state: np.ndarray | None
    steps_run: int
    converged: bool
    mean_phase_delta: float

