"""Test the ReLU activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.relu import ReLUOperation


class TestReLUOperation(unittest.TestCase):
    """Test the ReLU activation operation"""

    def test_forward_mixed_values(self):
        """Test the forward pass with mixed positive and negative values

        It should return x for positive values and 0 for negative values.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_all_positive(self):
        """Test the forward pass with all positive values

        It should return the input unchanged.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([0.1, 1.0, 2.0, 5.0])
        result = operation.forward(x=input_tensor)

        self.assertTrue(torch.allclose(result, input_tensor))

    def test_forward_all_negative(self):
        """Test the forward pass with all negative values

        It should return all zeros.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([-2.0, -1.0, -0.5])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([0.0, 0.0, 0.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_zeros(self):
        """Test the forward pass with zeros

        It should return zeros unchanged.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([0.0, 0.0, 0.0])
        result = operation.forward(x=input_tensor)

        self.assertTrue(torch.allclose(result, input_tensor))

    def test_forward_multidimensional_tensor(self):
        """Test the forward pass with multidimensional tensors

        It should apply ReLU element-wise to all dimensions.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([[[0.0, 2.0], [3.0, 0.0]], [[0.0, 6.0], [7.0, 0.0]]])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_different_dtypes(self):
        """Test the forward pass with different tensor dtypes

        It should work with float16, float32, and float64.
        """
        operation = ReLUOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
                result = operation.forward(x=input_tensor)

                # Check that output has same dtype
                self.assertEqual(result.dtype, dtype)

                # Check values are correct (allowing for precision differences in float16)
                if dtype == torch.float16:
                    self.assertTrue(torch.allclose(result, torch.tensor([0.0, 0.0, 1.0], dtype=dtype), rtol=1e-3))
                else:
                    expected = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
                    self.assertTrue(torch.allclose(result, expected))

    def test_forward_requires_grad(self):
        """Test that the operation preserves gradient requirements

        The output should require gradients if input requires gradients.
        """
        operation = ReLUOperation()

        input_tensor = torch.tensor([-1.0, 1.0], requires_grad=True)
        result = operation.forward(x=input_tensor)

        self.assertTrue(result.requires_grad)

        # Test backward pass
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)

    def test_forward_gradient_flow(self):
        """Test gradient flow through ReLU

        Gradients should flow for positive inputs but not for negative inputs.
        """
        operation = ReLUOperation()

        # Test with positive input
        x_pos = torch.tensor([1.0], requires_grad=True)
        y_pos = operation.forward(x=x_pos)
        y_pos.backward()
        assert x_pos.grad is not None
        self.assertEqual(x_pos.grad.item(), 1.0)

        # Test with negative input (gradient should be zero due to ReLU)
        x_neg = torch.tensor([-1.0], requires_grad=True)
        y_neg = operation.forward(x=x_neg)
        y_neg.backward()
        assert x_neg.grad is not None
        self.assertEqual(x_neg.grad.item(), 0.0)

    def test_forward_large_tensor(self):
        """Test the forward pass with a large tensor

        It should handle tensors of various sizes efficiently.
        """
        operation = ReLUOperation()

        # Create a large tensor with mixed values
        input_tensor = torch.randn(100, 50, 25) - 0.5  # Mean around -0.5, so roughly half negative
        result = operation.forward(x=input_tensor)

        # Check that all negative values became zero
        self.assertTrue(torch.all(result >= 0))

        # Check that positive values remained unchanged
        positive_mask = input_tensor > 0
        self.assertTrue(torch.allclose(result[positive_mask], input_tensor[positive_mask]))

        # Check that negative values became zero
        negative_mask = input_tensor <= 0
        self.assertTrue(torch.all(result[negative_mask] == 0))

    def test_forward_in_place_modification(self):
        """Test that the operation doesn't modify input tensor in-place

        The input tensor should remain unchanged after the forward pass.
        """
        operation = ReLUOperation()

        original_tensor = torch.tensor([-1.0, 0.0, 1.0])
        input_tensor = original_tensor.clone()

        result = operation.forward(x=input_tensor)

        # Input should be unchanged
        self.assertTrue(torch.allclose(input_tensor, original_tensor))

        # Result should be different from input
        self.assertFalse(torch.allclose(result, input_tensor))