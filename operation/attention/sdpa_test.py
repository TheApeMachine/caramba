"""Test the SDPA operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.attention.sdpa import SDPAOperation


class TestSDPAOperation(unittest.TestCase):
    """Test the SDPA operation."""

    def test_runs_and_shapes(self) -> None:
        op = SDPAOperation(dropout_p=0.0, is_causal=False)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = op.forward(q=q, k=k, v=v)
        self.assertEqual(out.shape, q.shape)

    def test_bool_mask_semantics_true_means_masked(self) -> None:
        op = SDPAOperation(dropout_p=0.0, is_causal=False)
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        v = torch.randn(1, 1, 2, 4)

        # Mask out everything -> output should be all zeros (no valid attention).
        mask_all = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        out_masked = op.forward(q=q, k=k, v=v, mask=mask_all)
        self.assertTrue(torch.allclose(out_masked, torch.zeros_like(out_masked)))

