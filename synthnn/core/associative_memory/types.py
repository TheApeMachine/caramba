"""Associative memory types

Defines result payloads produced by recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


SelectionMode: TypeAlias = Literal["full", "rerank(masked_final->full)"]


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Recall result

    Per-field meanings:
    - **label** (`str | None`): Selected label (from stored labels or generated via `label_prefix`).
    - **index** (`int | None`): Index of the selected pattern in the stored pattern matrix.
    - **score** (`float`): Similarity score for the selected pattern (higher is better).
    - **scores** (`np.ndarray`): 1D float array of per-pattern scores (shape `(K,)`).
    - **masked_scores** (`np.ndarray | None`): Optional 1D float array (shape `(K,)`) used during reranking for partial cues.
    - **selection** (`SelectionMode`): How the winner was chosen (full scoring vs rerank path).
    - **final_state** (`np.ndarray`): Final settled complex state (shape `(N,)`, complex dtype).
    - **snapped_state** (`np.ndarray | None`): Optional snapped pattern aligned by a global phase (shape `(N,)`, complex dtype).
    - **steps_run** (`int`): Number of dynamics steps executed.
    - **converged** (`bool`): Whether convergence criteria were met.
    - **mean_phase_delta** (`float`): Mean absolute phase delta between successive projected states.
    """

    label: str | None
    index: int | None
    score: float
    scores: np.ndarray
    masked_scores: np.ndarray | None
    selection: SelectionMode
    final_state: np.ndarray
    snapped_state: np.ndarray | None
    steps_run: int
    converged: bool
    mean_phase_delta: float

