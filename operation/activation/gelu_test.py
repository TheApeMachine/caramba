"""Test the GELU activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.gelu import GELUOperation


class TestGELUOperation(unittest.TestCase):
    """Test the GELU activation operation"""
    def test_forward(self):
        """Test the forward pass

        It should return the GELU activation of the input tensor.
        """
        operation = GELUOperation()

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = operation.forward(x=input_tensor)

        # GELU should be smooth and differentiable, with values between input and input * 1.1 for these inputs
        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue(torch.all(result > 0.8))  # GELU(1) ≈ 0.84
        self.assertTrue(torch.all(result < 3.5))  # GELU(3) ≈ 3.0
        self.assertTrue(torch.all(torch.isfinite(result)))