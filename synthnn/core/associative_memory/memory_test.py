from __future__ import annotations

from pathlib import Path

import numpy as np

from synthnn.core.associative_memory.memory import PhaseAssociativeMemory


class TestPhaseAssociativeMemory:
    """Tests for PhaseAssociativeMemory."""

    def test_store_and_recall_noisy(self) -> None:
        rng = np.random.default_rng(123)
        n = 32
        k = 8
        angles = rng.uniform(-np.pi, np.pi, size=(k, n)).astype(np.float64)
        patterns = np.exp(1j * angles)
        labels = [f"p{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(n, dtype=np.dtype(np.complex128), label_prefix=None)
        mem.store(patterns, labels=labels)

        target = 3
        base = patterns[target]
        cue = np.exp(1j * (np.angle(base) + rng.normal(0.0, 0.4, size=n)))
        res = mem.recall(cue, steps=200, dt=0.05, snap=True)
        assert res.index is not None
        assert res.label is not None
        assert res.label == labels[target]

    def test_store_and_recall_partial_with_rerank(self) -> None:
        rng = np.random.default_rng(7)
        n = 48
        k = 10
        patterns = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(k, n)))
        labels = [f"p{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(n, dtype=np.dtype(np.complex64), label_prefix="p")
        mem.store(patterns, labels=labels)

        target = 6
        base = patterns[target]
        mask = rng.random(n) < 0.25
        cue = base.copy()
        cue[~mask] = 1.0 + 0.0j
        res = mem.recall(cue, mask=mask, steps=250, dt=0.05, snap=True, rerank_top_k=5)
        assert res.label == labels[target]
        assert res.selection in {"full", "rerank(masked_final->full)"}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 16
        k = 4
        patterns = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(k, n)))
        labels = [f"p{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(n, dtype=np.dtype(np.complex128))
        mem.store(patterns, labels=labels)

        path = tmp_path / "mem.npz"
        mem.save(str(path))
        mem2 = PhaseAssociativeMemory.load(str(path))

        cue = patterns[1]
        r1 = mem.recall(cue, steps=50, dt=0.05)
        r2 = mem2.recall(cue, steps=50, dt=0.05)
        assert r1.label == r2.label

