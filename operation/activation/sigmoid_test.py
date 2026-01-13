"""Test the Sigmoid activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.sigmoid import SigmoidOperation


class TestSigmoidOperation(unittest.TestCase):
    """Test the Sigmoid activation operation"""

    def test_forward_basic_values(self):
        """Test the forward pass with basic values

        It should map inputs to the (0, 1) range using the sigmoid function.
        """
        operation = SigmoidOperation()

        input_tensor = torch.tensor([0.0, 1.0, -1.0])
        result = operation.forward(x=input_tensor)

        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
        expected = torch.tensor([0.5, 0.7310585786300049, 0.2689414213699951])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_range_validation(self):
        """Test that sigmoid outputs are always in (0, 1) range

        All outputs should be strictly greater than 0 and less than 1.
        """
        operation = SigmoidOperation()

        # Test with extreme values
        input_tensor = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        result = operation.forward(x=input_tensor)

        # All values should be in (0, 1)
        self.assertTrue(torch.all(result > 0))
        self.assertTrue(torch.all(result < 1))

        # Check specific known values
        self.assertAlmostEqual(result[3].item(), 0.5, places=6)  # sigmoid(0) = 0.5

    def test_forward_symmetry(self):
        """Test sigmoid symmetry around zero

        sigmoid(-x) + sigmoid(x) should equal 1.
        """
        operation = SigmoidOperation()

        x = torch.tensor([0.5, 1.0, 2.0, 3.0])
        neg_x = -x

        result_pos = operation.forward(x=x)
        result_neg = operation.forward(x=neg_x)

        # sigmoid(-x) + sigmoid(x) = 1
        sum_result = result_pos + result_neg

        self.assertTrue(torch.allclose(sum_result, torch.ones_like(sum_result)))

    def test_forward_large_positive_values(self):
        """Test sigmoid with large positive values

        It should approach 1.
        """
        operation = SigmoidOperation()

        large_values = torch.tensor([5.0, 10.0])
        result = operation.forward(x=large_values)

        # All should be very close to 1
        self.assertTrue(torch.all(result > 0.99))
        # Note: Very large values may reach exactly 1.0 due to numerical precision

    def test_forward_large_negative_values(self):
        """Test sigmoid with large negative values

        It should approach 0 but never reach it.
        """
        operation = SigmoidOperation()

        large_values = torch.tensor([-5.0, -10.0, -20.0])
        result = operation.forward(x=large_values)

        # All should be very close to 0 but greater than 0
        self.assertTrue(torch.all(result > 0.0))
        self.assertTrue(torch.all(result < 0.01))

    def test_forward_zeros(self):
        """Test sigmoid with zero input

        sigmoid(0) should equal 0.5.
        """
        operation = SigmoidOperation()

        input_tensor = torch.tensor([0.0])
        result = operation.forward(x=input_tensor)

        self.assertAlmostEqual(result.item(), 0.5, places=6)

    def test_forward_multidimensional_tensor(self):
        """Test the forward pass with multidimensional tensors

        It should apply sigmoid element-wise to all dimensions.
        """
        operation = SigmoidOperation()

        input_tensor = torch.tensor([[[-1.0, 0.0], [1.0, 2.0]], [[-2.0, 3.0], [0.5, -0.5]]])
        result = operation.forward(x=input_tensor)

        # Check that all values are in (0, 1)
        self.assertTrue(torch.all(result > 0))
        self.assertTrue(torch.all(result < 1))

        # Check specific values
        self.assertAlmostEqual(result[0, 0, 1].item(), 0.5, places=6)  # sigmoid(0) = 0.5

    def test_forward_different_dtypes(self):
        """Test the forward pass with different tensor dtypes

        It should work with float16, float32, and float64.
        """
        operation = SigmoidOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor([0.0, 1.0, -1.0], dtype=dtype)
                result = operation.forward(x=input_tensor)

                # Check that output has same dtype
                self.assertEqual(result.dtype, dtype)

                # Check that sigmoid(0) = 0.5 (allowing for precision differences in float16)
                if dtype == torch.float16:
                    self.assertAlmostEqual(result[0].item(), 0.5, places=3)
                else:
                    self.assertAlmostEqual(result[0].item(), 0.5, places=6)

    def test_forward_requires_grad(self):
        """Test that the operation preserves gradient requirements

        The output should require gradients if input requires gradients.
        """
        operation = SigmoidOperation()

        input_tensor = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        result = operation.forward(x=input_tensor)

        self.assertTrue(result.requires_grad)

        # Test backward pass
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)

    def test_forward_gradient_flow(self):
        """Test gradient flow through sigmoid

        Gradients should be computed correctly.
        """
        operation = SigmoidOperation()

        x = torch.tensor([0.0], requires_grad=True)
        y = operation.forward(x=x)
        y.backward()

        assert x.grad is not None
        # Gradient of sigmoid at 0 should be sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        expected_grad = 0.5 * (1 - 0.5)  # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=5)

    def test_forward_numerical_stability(self):
        """Test numerical stability with extreme values

        Sigmoid should handle very large and very small values without overflow.
        """
        operation = SigmoidOperation()

        # Test with very large values
        very_large = torch.tensor([100.0, 1000.0])
        result_large = operation.forward(x=very_large)

        # Should approach 1.0 but not overflow
        self.assertTrue(torch.all(result_large > 0.999))
        self.assertTrue(torch.all(result_large <= 1.0))
        self.assertFalse(torch.any(torch.isnan(result_large)))
        self.assertFalse(torch.any(torch.isinf(result_large)))

        # Test with very small values
        very_small = torch.tensor([-100.0, -1000.0])
        result_small = operation.forward(x=very_small)

        # Should approach 0.0 but not underflow
        self.assertTrue(torch.all(result_small >= 0.0))
        self.assertTrue(torch.all(result_small < 0.001))
        self.assertFalse(torch.any(torch.isnan(result_small)))
        self.assertFalse(torch.any(torch.isinf(result_small)))