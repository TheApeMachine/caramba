"""Test the CausalMaskLike operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.masking.causal_mask_like import CausalMaskLikeOperation


class TestCausalMaskLikeOperation(unittest.TestCase):
    """Test dynamic causal masking."""

    def test_builds_rectangular_mask(self) -> None:
        op = CausalMaskLikeOperation(diagonal=1)
        q = torch.randn(2, 4, 8)  # (B, T_q, D)
        k = torch.randn(2, 5, 8)  # (B, T_k, D)
        mask = op.forward(q=q, k=k)
        self.assertEqual(mask.shape, (1, 1, 4, 5))
        # For diagonal=1, mask[i,j] is True when j > i.
        m = mask[0, 0]
        self.assertFalse(bool(m[0, 0]))
        self.assertTrue(bool(m[0, 1]))
        self.assertTrue(bool(m[0, 4]))

    def test_diagonal_offset(self) -> None:
        op = CausalMaskLikeOperation(diagonal=2)
        q = torch.randn(1, 3, 4)
        k = torch.randn(1, 4, 4)  # e.g. null token prepended
        mask = op.forward(q=q, k=k)[0, 0]
        # Query row 0 allows keys 0..1; key 2 should be masked.
        self.assertFalse(bool(mask[0, 1]))
        self.assertTrue(bool(mask[0, 2]))

