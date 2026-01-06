"""Tests for MOSAIC memory VSA hybrid (B + D)

Verifies that:
- VSA tags can bias in-bucket selection without changing bucket routing.
- Novelty soft-scales writes when a candidate tag is redundant.
"""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from caramba.layer.mosaic.memory.reader import MemoryReader
from caramba.layer.mosaic.memory.phase import PhaseSimilarity, PhaseTagProjector
from caramba.layer.mosaic.memory.vsa import VsaNovelty, VsaTagProjector
from caramba.layer.mosaic.memory.writer import MemoryWriter
from caramba.layer.mosaic.state import MosaicState


class MemoryVsaHybridTest(unittest.TestCase):
    """Unit tests for VSA tag + novelty integration."""

    def test_vsa_tag_biases_slot_selection(self) -> None:
        torch.manual_seed(0)
        B, T, D = 1, 1, 8
        key_dim = 4
        tag_dim = 4
        mem_dim = 6
        hashes = 1
        buckets = 2
        assoc = 2

        mem_out = nn.Linear(mem_dim, D, bias=False)
        mem_qkey = nn.Linear(D, key_dim, bias=False)
        projector = VsaTagProjector(in_dim=key_dim, vsa_dim=tag_dim, tanh_scale=1.0)
        reader = MemoryReader(
            mem_out=mem_out,
            mem_qkey=mem_qkey,
            mem_key_dim=key_dim,
            mem_dim=mem_dim,
            mem_tag_dim=tag_dim,
            mem_assoc=assoc,
            mem_hashes=hashes,
            mem_buckets=buckets,
            mem_table_buckets=buckets,
            mem_read_temp=0.25,
            mem_vsa_weight=2.0,
            mem_vsa_enabled=True,
            vsa_projector=projector,
            mem_phase_weight=0.0,
            mem_phase_enabled=False,
            phase_projector=None,
            phase_similarity=None,
            mem_trie_enabled=False,
            mem_trie_fallback_enabled=False,
            mem_trie_max_levels=None,
        )

        u = torch.randn((B, T, D))
        qk = mem_qkey(u)
        qt = projector(qk)

        mem_k = torch.zeros((B, hashes, buckets, assoc, key_dim))
        mem_v = torch.zeros((B, hashes, buckets, assoc, mem_dim))
        mem_tag = torch.zeros((B, hashes, buckets, assoc, tag_dim))
        mem_last = torch.full((B, hashes, buckets, assoc), -1, dtype=torch.long)

        bucket = 0
        mem_last[0, 0, bucket, 0] = 0
        mem_last[0, 0, bucket, 1] = 0
        mem_k[0, 0, bucket, 0, :] = qk[0, 0, :]
        mem_k[0, 0, bucket, 1, :] = qk[0, 0, :]
        mem_tag[0, 0, bucket, 0, :] = qt[0, 0, :]
        mem_tag[0, 0, bucket, 1, :] = -qt[0, 0, :]
        mem_v[0, 0, bucket, 0, 0] = 1.0
        mem_v[0, 0, bucket, 1, 0] = -1.0

        st = MosaicState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_tag=mem_tag,
            mem_last=mem_last,
        )
        routing = {"idx_r": torch.zeros((B, T, hashes), dtype=torch.long), "collect_aux": True}

        _ = reader.read(u, st, routing)
        w = routing["read_slot_weights"]
        self.assertEqual(tuple(w.shape), (B, hashes, T, assoc))
        self.assertGreater(float(w[0, 0, 0, 0]), float(w[0, 0, 0, 1]))

    def test_novelty_soft_scales_writes(self) -> None:
        torch.manual_seed(0)
        B, T, D = 1, 1, 8
        key_dim = 4
        tag_dim = 4
        mem_dim = 6
        hashes = 1
        buckets = 2
        assoc = 2

        mem_wkey = nn.Linear(D, key_dim, bias=False)
        mem_value = nn.Linear(D, mem_dim, bias=False)
        mem_write_gate = nn.Linear(D, 1, bias=True)
        with torch.no_grad():
            mem_write_gate.bias.fill_(10.0)

        projector = VsaTagProjector(in_dim=key_dim, vsa_dim=tag_dim, tanh_scale=1.0)
        novelty = VsaNovelty(beta=10.0, threshold=0.0)
        writer = MemoryWriter(
            mem_wkey=mem_wkey,
            mem_value=mem_value,
            mem_write_gate=mem_write_gate,
            mem_buckets=buckets,
            mem_table_buckets=buckets,
            mem_hashes=hashes,
            mem_assoc=assoc,
            mem_key_dim=key_dim,
            mem_tag_dim=tag_dim,
            mem_dim=mem_dim,
            mem_write_threshold=0.0,
            mem_write_eta=1.0,
            mem_match_threshold=0.0,
            mem_vsa_enabled=True,
            vsa_projector=projector,
            vsa_novelty=novelty,
            mem_trie_enabled=False,
            mem_trie_eta_decay=0.5,
            mem_trie_max_levels=None,
        )

        u = torch.randn((B, T, D))
        wk = mem_wkey(u)
        wt = projector(wk)

        mem_k = torch.zeros((B, hashes, buckets, assoc, key_dim))
        mem_v = torch.zeros((B, hashes, buckets, assoc, mem_dim))
        mem_tag = torch.zeros((B, hashes, buckets, assoc, tag_dim))
        mem_last = torch.full((B, hashes, buckets, assoc), -1, dtype=torch.long)
        mem_last[0, 0, 0, 0] = 0
        mem_tag[0, 0, 0, 0, :] = wt[0, 0, :]

        st = MosaicState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_tag=mem_tag,
            mem_last=mem_last,
        )
        routing = {"idx_w": torch.zeros((B, T, hashes), dtype=torch.long), "collect_aux": True}

        gate_logits = writer.write_chunk(u, st, routing, 0, None, None)
        self.assertIsInstance(gate_logits, Tensor)
        wnov = routing["write_novelty"]
        self.assertEqual(tuple(wnov.shape), (B, T))
        self.assertLess(float(wnov[0, 0]), 0.5)

    def test_phase_similarity_is_global_phase_invariant(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        B, T, H, A = 1, 1, 1, 2
        key_dim = 4
        phase_dim = 8

        proj = PhaseTagProjector(in_dim=key_dim, phase_dim=phase_dim, tanh_scale=1.0).to(device)
        sim = PhaseSimilarity(phase_dim=phase_dim).to(device)

        qk = torch.randn((B, T, key_dim), device=device)
        bk = torch.randn((B, H, T, A, key_dim), device=device)
        valid = torch.ones((B, H, T, A), device=device, dtype=torch.bool)

        qphi = proj(qk)
        kphi = proj(bk)
        score1 = sim.score(q_angles=qphi, k_angles=kphi, valid=valid, batch=B, time=T)

        delta = 0.7
        score2 = sim.score(q_angles=qphi + delta, k_angles=kphi + delta, valid=valid, batch=B, time=T)
        self.assertTrue(torch.allclose(score1, score2, atol=1e-5, rtol=1e-5))

    def test_phase_scoring_can_bias_slot_selection(self) -> None:
        torch.manual_seed(0)
        B, T, D = 1, 1, 8
        key_dim = 4
        phase_dim = 8
        tag_dim = 4
        mem_dim = 6
        hashes = 1
        buckets = 2
        assoc = 2

        mem_out = nn.Linear(mem_dim, D, bias=False)
        mem_qkey = nn.Linear(D, key_dim, bias=False)
        phase_proj = PhaseTagProjector(in_dim=key_dim, phase_dim=phase_dim, tanh_scale=1.0)
        phase_sim = PhaseSimilarity(phase_dim=phase_dim)
        reader = MemoryReader(
            mem_out=mem_out,
            mem_qkey=mem_qkey,
            mem_key_dim=key_dim,
            mem_dim=mem_dim,
            mem_tag_dim=tag_dim,
            mem_assoc=assoc,
            mem_hashes=hashes,
            mem_buckets=buckets,
            mem_table_buckets=buckets,
            mem_read_temp=0.25,
            mem_vsa_weight=0.0,
            mem_vsa_enabled=False,
            vsa_projector=None,
            mem_phase_weight=5.0,
            mem_phase_enabled=True,
            phase_projector=phase_proj,
            phase_similarity=phase_sim,
            mem_trie_enabled=False,
            mem_trie_fallback_enabled=False,
            mem_trie_max_levels=None,
        )

        u = torch.randn((B, T, D))
        qk = mem_qkey(u)

        mem_k = torch.zeros((B, hashes, buckets, assoc, key_dim))
        mem_v = torch.zeros((B, hashes, buckets, assoc, mem_dim))
        mem_tag = torch.zeros((B, hashes, buckets, assoc, tag_dim))
        mem_last = torch.full((B, hashes, buckets, assoc), -1, dtype=torch.long)

        bucket = 0
        mem_last[0, 0, bucket, 0] = 0
        mem_last[0, 0, bucket, 1] = 0
        mem_k[0, 0, bucket, 0, :] = qk[0, 0, :]
        mem_k[0, 0, bucket, 1, :] = -qk[0, 0, :]
        mem_v[0, 0, bucket, 0, 0] = 1.0
        mem_v[0, 0, bucket, 1, 0] = -1.0

        st = MosaicState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_tag=mem_tag,
            mem_last=mem_last,
        )
        routing = {"idx_r": torch.zeros((B, T, hashes), dtype=torch.long), "collect_aux": True}
        _ = reader.read(u, st, routing)
        w = routing["read_slot_weights"]
        self.assertGreater(float(w[0, 0, 0, 0]), float(w[0, 0, 0, 1]))

    def test_trie_fallback_reads_parent_prefix(self) -> None:
        torch.manual_seed(0)
        B, T, D = 1, 1, 4
        key_dim = 4
        mem_dim = 4
        hashes = 1
        leaves = 4
        table = 2 * leaves - 1  # 7
        assoc = 1

        mem_out = nn.Linear(mem_dim, D, bias=False)
        with torch.no_grad():
            mem_out.weight.copy_(torch.eye(D))
        mem_qkey = nn.Linear(D, key_dim, bias=False)
        reader = MemoryReader(
            mem_out=mem_out,
            mem_qkey=mem_qkey,
            mem_key_dim=key_dim,
            mem_dim=mem_dim,
            mem_tag_dim=1,
            mem_assoc=assoc,
            mem_hashes=hashes,
            mem_buckets=leaves,
            mem_table_buckets=table,
            mem_read_temp=1.0,
            mem_vsa_weight=0.0,
            mem_vsa_enabled=False,
            vsa_projector=None,
            mem_phase_weight=0.0,
            mem_phase_enabled=False,
            phase_projector=None,
            phase_similarity=None,
            mem_trie_enabled=True,
            mem_trie_fallback_enabled=True,
            mem_trie_max_levels=None,
        )

        u = torch.randn((B, T, D))
        qk = mem_qkey(u)

        mem_k = torch.zeros((B, hashes, table, assoc, key_dim))
        mem_v = torch.zeros((B, hashes, table, assoc, mem_dim))
        mem_tag = torch.zeros((B, hashes, table, assoc, 1))
        mem_last = torch.full((B, hashes, table, assoc), -1, dtype=torch.long)

        leaf = 2
        leaf_node = (leaves - 1) + leaf  # base + leaf
        parent = (leaf_node - 1) // 2
        # Leaf is empty; parent holds the value.
        mem_last[0, 0, parent, 0] = 0
        mem_k[0, 0, parent, 0, :] = qk[0, 0, :]
        mem_v[0, 0, parent, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        st = MosaicState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_tag=mem_tag,
            mem_last=mem_last,
        )
        routing = {"idx_r": torch.full((B, T, hashes), leaf, dtype=torch.long), "collect_aux": True}
        y = reader.read(u, st, routing)  # (B,T,D)
        self.assertTrue("trie_fallback_steps" in routing)
        steps = routing["trie_fallback_steps"]
        self.assertEqual(int(steps[0, 0, 0]), 1)
        self.assertTrue(torch.allclose(y[0, 0], mem_v[0, 0, parent, 0], atol=1e-5, rtol=1e-5))

    def test_trie_ancestor_write_scales_eta(self) -> None:
        torch.manual_seed(0)
        B, T, D = 1, 1, 4
        key_dim = 4
        mem_dim = 4
        hashes = 1
        leaves = 4
        table = 2 * leaves - 1
        assoc = 1

        mem_wkey = nn.Linear(D, key_dim, bias=False)
        mem_value = nn.Linear(D, mem_dim, bias=False)
        mem_write_gate = nn.Linear(D, 1, bias=True)
        with torch.no_grad():
            mem_write_gate.bias.fill_(10.0)

        writer = MemoryWriter(
            mem_wkey=mem_wkey,
            mem_value=mem_value,
            mem_write_gate=mem_write_gate,
            mem_buckets=leaves,
            mem_table_buckets=table,
            mem_hashes=hashes,
            mem_assoc=assoc,
            mem_key_dim=key_dim,
            mem_tag_dim=1,
            mem_dim=mem_dim,
            mem_write_threshold=0.0,
            mem_write_eta=1.0,
            mem_match_threshold=0.0,
            mem_vsa_enabled=False,
            vsa_projector=None,
            vsa_novelty=None,
            mem_trie_enabled=True,
            mem_trie_eta_decay=0.5,
            mem_trie_max_levels=1,  # leaf + parent only
        )

        u = torch.randn((B, T, D))
        wk = mem_wkey(u)
        v = mem_value(u)

        mem_k = torch.zeros((B, hashes, table, assoc, key_dim))
        mem_vt = torch.zeros((B, hashes, table, assoc, mem_dim))
        mem_tag = torch.zeros((B, hashes, table, assoc, 1))
        mem_last = torch.full((B, hashes, table, assoc), -1, dtype=torch.long)

        leaf = 2
        leaf_node = (leaves - 1) + leaf
        parent = (leaf_node - 1) // 2
        # Pre-fill leaf and parent so writes take the "update" path and eta matters.
        mem_last[0, 0, leaf_node, 0] = 0
        mem_last[0, 0, parent, 0] = 0
        mem_k[0, 0, leaf_node, 0, :] = wk[0, 0, :]
        mem_k[0, 0, parent, 0, :] = wk[0, 0, :]

        st = MosaicState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=mem_k,
            mem_v=mem_vt,
            mem_tag=mem_tag,
            mem_last=mem_last,
        )
        routing = {"idx_w": torch.full((B, T, hashes), leaf, dtype=torch.long), "collect_aux": False}
        gate_logits = writer.write_chunk(u, st, routing, 0, None, None)
        p = torch.sigmoid(gate_logits.detach())[0, 0].to(dtype=v.dtype)

        leaf_written = st.mem_v[0, 0, leaf_node, 0, :]
        parent_written = st.mem_v[0, 0, parent, 0, :]
        self.assertTrue(torch.allclose(leaf_written, p * v[0, 0], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(parent_written, (0.5 * p) * v[0, 0], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

