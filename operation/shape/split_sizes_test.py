"""Test the SplitSizes operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.split_sizes import SplitSizesOperation


class TestSplitSizesOperation(unittest.TestCase):
    """Test splitting tensors by explicit sizes."""

    def test_basic_split_sizes(self) -> None:
        operation = SplitSizesOperation(split_sizes=[2, 3, 1], dim=-1)
        x = torch.randn(4, 6)
        a, b, c = operation.forward(x=x)
        self.assertEqual(a.shape, (4, 2))
        self.assertEqual(b.shape, (4, 3))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue(torch.allclose(torch.cat([a, b, c], dim=-1), x))

    def test_negative_dim(self) -> None:
        operation = SplitSizesOperation(split_sizes=[1, 2], dim=-3)
        x = torch.randn(3, 4, 2)
        a, b = operation.forward(x=x)
        self.assertEqual(a.shape, (1, 4, 2))
        self.assertEqual(b.shape, (2, 4, 2))

    def test_requires_non_empty_sizes(self) -> None:
        with self.assertRaises(ValueError) as context:
            SplitSizesOperation(split_sizes=[])
        self.assertIn("split_sizes must be non-empty", str(context.exception))

    def test_requires_positive_sizes(self) -> None:
        with self.assertRaises(ValueError) as context:
            SplitSizesOperation(split_sizes=[2, 0, 1])
        self.assertIn("split_sizes must contain only positive ints", str(context.exception))

    def test_sum_must_match_dim(self) -> None:
        operation = SplitSizesOperation(split_sizes=[2, 2], dim=-1)
        x = torch.randn(1, 5)
        with self.assertRaises(ValueError) as context:
            _ = operation.forward(x=x)
        self.assertIn("split_sizes sum", str(context.exception))

    def test_gradients_flow(self) -> None:
        operation = SplitSizesOperation(split_sizes=[2, 2], dim=-1)
        x = torch.randn(3, 4, requires_grad=True)
        a, b = operation.forward(x=x)
        loss = a.sum() + b.sum()
        loss.backward()
        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_preserves_dtype(self) -> None:
        operation = SplitSizesOperation(split_sizes=[1, 1], dim=-1)
        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 2, dtype=dtype)
                a, b = operation.forward(x=x)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(b.dtype, dtype)
