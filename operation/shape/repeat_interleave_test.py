"""Test the RepeatInterleave operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.repeat_interleave import RepeatInterleaveOperation


class TestRepeatInterleaveOperation(unittest.TestCase):
    """Test the RepeatInterleave operation."""

    def test_repeat_interleave_basic(self) -> None:
        op = RepeatInterleaveOperation(repeats=2, dim=0)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = op.forward(x=x)
        self.assertTrue(torch.allclose(y, torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])))

    def test_repeat_interleave_dim(self) -> None:
        op = RepeatInterleaveOperation(repeats=3, dim=1)
        x = torch.randn(2, 4, 5)
        y = op.forward(x=x)
        self.assertEqual(y.shape, (2, 12, 5))

    def test_repeat_interleave_validation(self) -> None:
        with self.assertRaises(ValueError) as context:
            RepeatInterleaveOperation(repeats=0, dim=0)
        self.assertIn("repeats must be > 0", str(context.exception))

