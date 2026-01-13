"""Test the Tanh activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.tanh import TanhOperation
from typing_extensions import override


class TestTanhOperation(unittest.TestCase):
    """Test the Tanh activation operation"""

    def test_forward_basic_values(self):
        """Test the forward pass with basic values

        It should map inputs to the (-1, 1) range using the tanh function.
        """
        operation = TanhOperation()

        input_tensor = torch.tensor([0.0, 1.0, -1.0])
        result = operation.forward(x=input_tensor)

        # tanh(0) = 0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
        expected = torch.tensor([0.0, 0.7615941559557649, -0.7615941559557649])

        self.assertTrue(torch.allclose(result, expected))

    def test_forward_range_validation(self):
        """Test that tanh outputs are always in (-1, 1) range

        All outputs should be strictly greater than -1 and less than 1.
        """
        operation = TanhOperation()

        # Test with extreme values
        input_tensor = torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        result = operation.forward(x=input_tensor)

        # All values should be in (-1, 1)
        self.assertTrue(torch.all(result > -1))
        self.assertTrue(torch.all(result < 1))

        # Check specific known values
        self.assertAlmostEqual(result[3].item(), 0.0, places=6)  # tanh(0) = 0

    def test_forward_zero_centering(self):
        """Test that tanh is zero-centered

        tanh(-x) should equal -tanh(x).
        """
        operation = TanhOperation()

        x = torch.tensor([0.5, 1.0, 2.0, 3.0])
        neg_x = -x

        result_pos = operation.forward(x=x)
        result_neg = operation.forward(x=neg_x)

        # tanh(-x) = -tanh(x)
        self.assertTrue(torch.allclose(result_neg, -result_pos))

    def test_forward_odd_function_property(self):
        """Test that tanh is an odd function

        This is equivalent to the zero-centering property.
        """
        operation = TanhOperation()

        x = torch.tensor([0.1, 0.5, 1.0, 2.0])
        result_x = operation.forward(x=x)
        result_neg_x = operation.forward(x=-x)

        # Verify tanh(-x) = -tanh(x)
        self.assertTrue(torch.allclose(result_neg_x, -result_x))

    def test_forward_large_positive_values(self):
        """Test tanh with large positive values

        It should approach 1.
        """
        operation = TanhOperation()

        large_values = torch.tensor([3.0, 5.0])
        result = operation.forward(x=large_values)

        # All should be very close to 1
        self.assertTrue(torch.all(result > 0.99))
        # Note: Very large values may reach exactly 1.0 due to numerical precision

    def test_forward_large_negative_values(self):
        """Test tanh with large negative values

        It should approach -1.
        """
        operation = TanhOperation()

        large_values = torch.tensor([-3.0, -5.0])
        result = operation.forward(x=large_values)

        # All should be very close to -1
        self.assertTrue(torch.all(result < -0.99))
        # Note: Very large negative values may reach exactly -1.0 due to numerical precision

    def test_forward_zeros(self):
        """Test tanh with zero input

        tanh(0) should equal 0.
        """
        operation = TanhOperation()

        input_tensor = torch.tensor([0.0])
        result = operation.forward(x=input_tensor)

        self.assertAlmostEqual(result.item(), 0.0, places=6)

    def test_forward_multidimensional_tensor(self):
        """Test the forward pass with multidimensional tensors

        It should apply tanh element-wise to all dimensions.
        """
        operation = TanhOperation()

        input_tensor = torch.tensor([[[-1.0, 0.0], [1.0, 2.0]], [[-2.0, 3.0], [0.5, -0.5]]])
        result = operation.forward(x=input_tensor)

        # Check that all values are in (-1, 1)
        self.assertTrue(torch.all(result > -1))
        self.assertTrue(torch.all(result < 1))

        # Check specific values
        self.assertAlmostEqual(result[0, 0, 1].item(), 0.0, places=6)  # tanh(0) = 0

    def test_forward_different_dtypes(self):
        """Test the forward pass with different tensor dtypes

        It should work with float16, float32, and float64.
        """
        operation = TanhOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor([0.0, 1.0, -1.0], dtype=dtype)
                result = operation.forward(x=input_tensor)

                # Check that output has same dtype
                self.assertEqual(result.dtype, dtype)

                # Check that tanh(0) = 0 (allowing for precision differences in float16)
                if dtype == torch.float16:
                    self.assertAlmostEqual(result[0].item(), 0.0, places=3)
                else:
                    self.assertAlmostEqual(result[0].item(), 0.0, places=6)

    def test_forward_requires_grad(self):
        """Test that the operation preserves gradient requirements

        The output should require gradients if input requires gradients.
        """
        operation = TanhOperation()

        input_tensor = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        result = operation.forward(x=input_tensor)

        self.assertTrue(result.requires_grad)

        # Test backward pass
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)

    def test_forward_gradient_flow(self):
        """Test gradient flow through tanh

        Gradients should be computed correctly.
        """
        operation = TanhOperation()

        x = torch.tensor([0.0], requires_grad=True)
        y = operation.forward(x=x)
        y.backward()

        assert x.grad is not None
        # Gradient of tanh at 0 should be 1 - tanh(0)^2 = 1 - 0 = 1
        expected_grad = 1.0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=5)

    def test_forward_gradient_saturation(self):
        """Test gradient behavior in saturation regions

        Gradients should approach zero for large |x| values.
        """
        operation = TanhOperation()

        # Test with large values where tanh approaches ±1
        x_large = torch.tensor([5.0], requires_grad=True)
        y_large = operation.forward(x=x_large)
        y_large.backward()

        assert x_large.grad is not None
        # Gradient should be very small (close to 0)
        self.assertLess(abs(x_large.grad.item()), 0.01)

        x_small = torch.tensor([-5.0], requires_grad=True)
        y_small = operation.forward(x=x_small)
        y_small.backward()

        assert x_small.grad is not None
        # Gradient should be very small (close to 0)
        self.assertLess(abs(x_small.grad.item()), 0.01)

    def test_forward_numerical_stability(self):
        """Test numerical stability with extreme values

        Tanh should handle very large and very small values without overflow.
        """
        operation = TanhOperation()

        # Test with very large values
        very_large = torch.tensor([100.0, 1000.0])
        result_large = operation.forward(x=very_large)

        # Should be very close to 1.0 but not overflow
        self.assertTrue(torch.all(result_large > 0.999))
        self.assertTrue(torch.all(result_large <= 1.0))
        self.assertFalse(torch.any(torch.isnan(result_large)))
        self.assertFalse(torch.any(torch.isinf(result_large)))

        # Test with very small values
        very_small = torch.tensor([-100.0, -1000.0])
        result_small = operation.forward(x=very_small)

        # Should be very close to -1.0 but not underflow
        self.assertTrue(torch.all(result_small >= -1.0))
        self.assertTrue(torch.all(result_small < -0.999))
        self.assertFalse(torch.any(torch.isnan(result_small)))
        self.assertFalse(torch.any(torch.isinf(result_small)))

    def test_forward_derivative_properties(self):
        """Test derivative properties of tanh

        The derivative of tanh(x) is sech^2(x) = 1 - tanh^2(x).
        """
        operation = TanhOperation()

        x = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        y = operation.forward(x=x)
        grad_outputs = torch.ones_like(y)
        grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]

        # Manual calculation of expected gradients: 1 - tanh(x)^2
        expected_grads = 1 - y**2

        self.assertTrue(torch.allclose(grads, expected_grads))

    def test_forward_identity_near_zero(self):
        """Test that tanh behaves like identity near zero

        For small x, tanh(x) ≈ x.
        """
        operation = TanhOperation()

        small_values = torch.tensor([-0.1, -0.01, 0.01, 0.1])
        result = operation.forward(x=small_values)

        # For small values, tanh(x) should be approximately equal to x
        self.assertTrue(torch.allclose(result, small_values, rtol=0.01))