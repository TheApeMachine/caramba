"""Test the LeakyReLU activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.leaky_relu import LeakyReLUOperation


class TestLeakyReLUOperation(unittest.TestCase):
    """Test the LeakyReLU activation operation"""

    def test_forward_default_negative_slope(self):
        """Test the forward pass with default negative slope

        It should return x for positive values and 0.01*x for negative values.
        """
        operation = LeakyReLUOperation()

        # Test with mixed positive and negative values
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([-0.02, -0.01, 0.0, 1.0, 2.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_custom_negative_slope(self):
        """Test the forward pass with custom negative slope

        It should return x for positive values and slope*x for negative values.
        """
        operation = LeakyReLUOperation(negative_slope=0.1)

        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_all_positive(self):
        """Test the forward pass with all positive values

        It should return the input unchanged.
        """
        operation = LeakyReLUOperation()

        input_tensor = torch.tensor([0.1, 1.0, 2.0, 5.0])
        result = operation.forward(x=input_tensor)

        self.assertTrue(torch.allclose(result, input_tensor))

    def test_forward_all_negative(self):
        """Test the forward pass with all negative values

        It should return negative_slope * input.
        """
        operation = LeakyReLUOperation(negative_slope=0.2)

        input_tensor = torch.tensor([-2.0, -1.0, -0.5])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([-0.4, -0.2, -0.1])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_zeros(self):
        """Test the forward pass with zeros

        It should return zeros unchanged.
        """
        operation = LeakyReLUOperation()

        input_tensor = torch.tensor([0.0, 0.0, 0.0])
        result = operation.forward(x=input_tensor)

        self.assertTrue(torch.allclose(result, input_tensor))

    def test_forward_multidimensional_tensor(self):
        """Test the forward pass with multidimensional tensors

        It should apply LeakyReLU element-wise to all dimensions.
        """
        operation = LeakyReLUOperation(negative_slope=0.01)

        input_tensor = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
        result = operation.forward(x=input_tensor)
        expected = torch.tensor([[[-0.01, 2.0], [3.0, -0.04]], [[-0.05, 6.0], [7.0, -0.08]]])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_different_dtypes(self):
        """Test the forward pass with different tensor dtypes

        It should work with float16, float32, and float64.
        """
        operation = LeakyReLUOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
                result = operation.forward(x=input_tensor)

                # Check that output has same dtype
                self.assertEqual(result.dtype, dtype)

                # Check values are correct (allowing for precision differences in float16)
                if dtype == torch.float16:
                    self.assertTrue(torch.allclose(result, torch.tensor([-0.01, 0.0, 1.0], dtype=dtype), rtol=1e-3))
                else:
                    expected = torch.tensor([-0.01, 0.0, 1.0], dtype=dtype)
                    self.assertTrue(torch.allclose(result, expected))

    def test_forward_requires_grad(self):
        """Test that the operation preserves gradient requirements

        The output should require gradients if input requires gradients.
        """
        operation = LeakyReLUOperation()

        input_tensor = torch.tensor([-1.0, 1.0], requires_grad=True)
        result = operation.forward(x=input_tensor)

        self.assertTrue(result.requires_grad)

        # Test backward pass
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)

    def test_negative_slope_parameter_validation(self):
        """Test that negative_slope parameter is stored correctly

        It should store the negative_slope parameter as an instance attribute.
        """
        operation = LeakyReLUOperation(negative_slope=0.05)
        self.assertEqual(operation.negative_slope, 0.05)

        operation_default = LeakyReLUOperation()
        self.assertEqual(operation_default.negative_slope, 0.01)

    def test_forward_gradient_flow(self):
        """Test gradient flow through LeakyReLU

        Gradients should flow properly for both positive and negative inputs.
        """
        operation = LeakyReLUOperation(negative_slope=0.1)

        # Test with positive input
        x_pos = torch.tensor([1.0], requires_grad=True)
        y_pos = operation.forward(x=x_pos)
        y_pos.backward()
        assert x_pos.grad is not None
        self.assertEqual(x_pos.grad.item(), 1.0)

        # Test with negative input
        x_neg = torch.tensor([-1.0], requires_grad=True)
        y_neg = operation.forward(x=x_neg)
        y_neg.backward()
        assert x_neg.grad is not None
        self.assertAlmostEqual(x_neg.grad.item(), 0.1, places=5)