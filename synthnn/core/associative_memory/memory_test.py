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
        patterns = np.exp(1j * angles).astype(np.complex128)
        labels = [f"p{i}" for i in range(k)]

        mem = PhaseAssociativeMemory(n, dtype=np.dtype(np.complex128), label_prefix=None)
        mem.store(patterns, labels=labels)

        target = 3
        base = patterns[target]
        cue = np.exp(1j * (np.angle(base) + rng.normal(0.0, 0.4, size=n)))
        res = mem.recall(cue, steps=200, dt=0.05, snap=True)
        assert res.index is not None
        assert res.label is not None
        assert res.index == target
        assert res.label == labels[target]
        assert res.converged is True or float(res.mean_phase_delta) < 0.25
        assert res.snapped_state is not None
        # Compare up to a global phase rotation.
        sim = float(np.abs(np.vdot(base, res.snapped_state)) / float(n))
        assert sim > 0.999

    def test_store_and_recall_partial_with_rerank(self) -> None:
        rng = np.random.default_rng(7)
        n = 48
        k = 10
        patterns = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(k, n)))
        # label_prefix should be responsible for producing labels when labels are not stored.
        labels = None
        mem = PhaseAssociativeMemory(n, dtype=np.dtype(np.complex64), label_prefix="p")
        mem.store(patterns, labels=labels)

        target = 6
        base = patterns[target]
        mask = rng.random(n) < 0.25
        cue = base.copy()
        cue[~mask] = 1.0 + 0.0j
        res = mem.recall(cue, mask=mask, steps=250, dt=0.05, snap=True, rerank_top_k=5)
        assert res.label == f"p{target}"
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

        assert path.exists()
        assert mem.num_units == mem2.num_units
        assert mem.dtype == mem2.dtype
        assert mem.label_prefix == mem2.label_prefix
        assert mem.coupling_strength == mem2.coupling_strength
        assert mem.damping == mem2.damping
        assert mem.zero_diag == mem2.zero_diag
        assert mem.clamp_cue == mem2.clamp_cue
        assert mem.project_each_step == mem2.project_each_step
        assert mem.project_interval == mem2.project_interval
        assert mem.projection_eps is not None
        assert mem2.projection_eps is not None
        assert float(mem.projection_eps) == float(mem2.projection_eps)
        assert mem.clamp_alpha == mem2.clamp_alpha
        assert mem.node_prefix == mem2.node_prefix
        assert mem.stored is not None and mem2.stored is not None
        assert np.allclose(mem.stored.patterns, mem2.stored.patterns)
        assert np.allclose(mem.stored.weights, mem2.stored.weights)
        assert mem.stored.labels == mem2.stored.labels

        cue = patterns[1]
        r1 = mem.recall(cue, steps=50, dt=0.05)
        r2 = mem2.recall(cue, steps=50, dt=0.05)
        assert r1.label == r2.label
        assert r1.index == r2.index
        assert np.allclose(r1.final_state, r2.final_state)
        if r1.snapped_state is not None and r2.snapped_state is not None:
            assert np.allclose(r1.snapped_state, r2.snapped_state)

