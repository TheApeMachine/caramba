"""Associative memory scoring

Scores settled states against stored patterns and performs optional reranking
for partial cues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.associative_memory.types import RecallResult, SelectionMode


@dataclass(frozen=True, slots=True)
class MemoryScorer:
    """Score and snap for phase associative memory."""

    num_units: int
    dtype: np.dtype
    projection_eps: float
    label_prefix: str | None

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError(f"MemoryScorer dtype must be complex, got {dtype}")

    def finish(
        self,
        *,
        patterns: np.ndarray,
        labels: list[str] | None,
        final_state: np.ndarray,
        known: np.ndarray,
        snap: bool,
        rerank_top_k: int,
        steps_run: int,
        converged: bool,
        mean_phase_delta: float,
    ) -> RecallResult:
        active = np.abs(final_state) > float(self.projection_eps)
        denom = float(np.sum(active)) if np.any(active) else float(self.num_units)

        v_full = final_state.copy()
        v_full[~active] = self.dtype.type(0.0 + 0.0j)
        dots = patterns @ np.conj(v_full)
        scores = np.abs(dots) / float(denom)

        masked_scores: np.ndarray | None = None
        selection: SelectionMode = "full"
        best_idx = int(np.argmax(scores)) if scores.size else None
        if int(rerank_top_k) > 0 and np.any(~known) and int(np.sum(known)) > 0:
            selection, best_idx, masked_scores = self.rerank(
                patterns=patterns,
                final_state=final_state,
                known=known,
                scores=scores,
                top_k=int(rerank_top_k),
            )

        label = self.labelOf(labels=labels, index=best_idx)
        best_score = float(scores[best_idx]) if best_idx is not None else 0.0
        if selection != "full" and masked_scores is not None and best_idx is not None:
            best_score = float(masked_scores[best_idx])

        snapped = self.snap(patterns=patterns, final_state=final_state, index=best_idx) if bool(snap) else None
        return RecallResult(
            label=label,
            index=best_idx,
            score=float(best_score),
            scores=scores,
            masked_scores=masked_scores,
            selection=selection,
            final_state=final_state,
            snapped_state=snapped,
            steps_run=int(steps_run),
            converged=bool(converged),
            mean_phase_delta=float(mean_phase_delta),
        )

    def rerank(
        self,
        *,
        patterns: np.ndarray,
        final_state: np.ndarray,
        known: np.ndarray,
        scores: np.ndarray,
        top_k: int,
    ) -> tuple[SelectionMode, int, np.ndarray]:
        m = int(np.sum(known))
        v_known = final_state.copy()
        v_known[~known] = self.dtype.type(0.0 + 0.0j)
        masked_dots = patterns @ np.conj(v_known)
        masked_scores = np.abs(masked_dots) / float(max(1, m))

        k = int(min(max(int(top_k), 1), int(masked_scores.size)))
        cand = np.argpartition(masked_scores, -k)[-k:]
        best_m = float(np.max(masked_scores[cand]))
        tie_tol = 1e-6 if self.dtype == np.dtype(np.complex64) else 1e-12
        tie = cand[masked_scores[cand] >= (best_m - float(tie_tol))]
        best_idx = int(tie[np.argmax(scores[tie])])
        return "rerank(masked_final->full)", best_idx, masked_scores

    def snap(self, *, patterns: np.ndarray, final_state: np.ndarray, index: int | None) -> np.ndarray | None:
        if index is None:
            return None
        p = patterns[int(index)]
        ph = np.vdot(p, final_state)
        # Treat near-zero inner products as zero to avoid noisy rotations.
        eps = float(np.finfo(np.dtype(np.float32)).eps)
        if np.isclose(np.abs(ph), 0.0, atol=eps, rtol=0.0):
            rot = self.dtype.type(1.0 + 0.0j)
        else:
            rot = np.exp(1j * np.angle(ph)).astype(self.dtype, copy=False)
        return (p * rot).astype(self.dtype, copy=False)

    def labelOf(self, *, labels: list[str] | None, index: int | None) -> str | None:
        if index is None:
            return None
        if labels is not None:
            return labels[int(index)]
        if self.label_prefix is None:
            return None
        return f"{self.label_prefix}{int(index)}"

