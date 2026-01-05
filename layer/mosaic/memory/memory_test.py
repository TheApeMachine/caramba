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
            mem_read_temp=0.25,
            mem_vsa_weight=2.0,
            mem_vsa_enabled=True,
            vsa_projector=projector,
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


if __name__ == "__main__":
    unittest.main()

