"""Test the Parameter operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.tensor.parameter import ParameterOperation


class TestParameterOperation(unittest.TestCase):
    """Test the Parameter operation."""

    def test_creates_parameter(self) -> None:
        op = ParameterOperation(shape=[2, 3], init="zeros")
        out = op()
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(torch.allclose(out, torch.zeros(2, 3)))

    def test_casts_like_input(self) -> None:
        op = ParameterOperation(shape=[1, 2], init="ones")
        like = torch.zeros(4, 2, dtype=torch.float16, device=torch.device("cpu"))
        out = op(like)
        self.assertEqual(out.dtype, like.dtype)
        self.assertEqual(out.device, like.device)

    def test_expand_batch(self) -> None:
        op = ParameterOperation(shape=[1, 3, 1, 4], init="zeros", expand_batch=True)
        like = torch.randn(2, 3, 5, 4)
        out = op(like)
        self.assertEqual(out.shape, (2, 3, 1, 4))

    def test_expand_batch_errors_for_wrong_dim0(self) -> None:
        op = ParameterOperation(shape=[2, 3], init="zeros", expand_batch=True)
        like = torch.randn(4, 3)
        with self.assertRaises(ValueError) as context:
            _ = op(like)
        self.assertIn("expand_batch requires parameter dim0==1", str(context.exception))

