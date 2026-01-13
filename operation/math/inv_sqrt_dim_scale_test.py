"""Test the InvSqrtDimScale operation"""
from __future__ import annotations

import math
import unittest

import torch

from caramba.operation.math.inv_sqrt_dim_scale import InvSqrtDimScaleOperation


class TestInvSqrtDimScaleOperation(unittest.TestCase):
    """Test the InvSqrtDimScale operation."""

    def test_scales_by_inverse_sqrt_last_dim(self) -> None:
        op = InvSqrtDimScaleOperation()
        x = torch.ones(2, 3, 4)
        y = op.forward(x=x)
        self.assertTrue(torch.allclose(y, x * (1.0 / math.sqrt(4.0))))

    def test_gradient_flow(self) -> None:
        op = InvSqrtDimScaleOperation()
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = op.forward(x=x)
        y.sum().backward()
        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

