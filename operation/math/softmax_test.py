"""Test the Softmax operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.softmax import SoftmaxOperation


class TestSoftmaxOperation(unittest.TestCase):
    """Test the Softmax operation"""

    def test_basic_softmax(self):
        """Test basic softmax normalization

        It should convert values to probabilities that sum to 1.
        """
        operation = SoftmaxOperation()

        x = torch.tensor([1.0, 2.0, 3.0])
        result = operation.forward(x=x)

        # Check that values are probabilities (between 0 and 1)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

        # Check that they sum to 1
        self.assertAlmostEqual(result.sum().item(), 1.0, places=6)

        # Verify against torch.softmax
        expected = torch.softmax(x, dim=-1)
        self.assertTrue(torch.allclose(result, expected))

    def test_softmax_multidimensional(self):
        """Test softmax on multidimensional tensors

        It should apply softmax along the last dimension by default.
        """
        operation = SoftmaxOperation()

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [2, 3]
        result = operation.forward(x=x)

        # Each row should sum to 1
        row_sums = result.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)))

        # All values should be between 0 and 1
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

    def test_softmax_custom_dimension(self):
        """Test softmax along custom dimension

        It should apply softmax along the specified dimension.
        """
        operation = SoftmaxOperation(dim=0)  # Along first dimension

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        result = operation.forward(x=x)

        # Each column should sum to 1
        col_sums = result.sum(dim=0)
        self.assertTrue(torch.allclose(col_sums, torch.ones_like(col_sums)))

    def test_softmax_uniform_input(self):
        """Test softmax with uniform input values

        It should produce uniform probabilities.
        """
        operation = SoftmaxOperation()

        x = torch.ones(5)  # All same values
        result = operation.forward(x=x)

        # All probabilities should be equal (1/5 = 0.2)
        expected_prob = 1.0 / 5.0
        self.assertTrue(torch.allclose(result, torch.full_like(result, expected_prob)))

    def test_softmax_single_element(self):
        """Test softmax with single element

        It should return [1.0].
        """
        operation = SoftmaxOperation()

        x = torch.tensor([2.5])
        result = operation.forward(x=x)

        self.assertEqual(result.item(), 1.0)

    def test_softmax_large_negative_values(self):
        """Test softmax with large negative values

        It should handle numerical stability correctly.
        """
        operation = SoftmaxOperation()

        x = torch.tensor([-1000.0, -999.0, -998.0])
        result = operation.forward(x=x)

        # Should still sum to 1 and be valid probabilities
        self.assertAlmostEqual(result.sum().item(), 1.0, places=6)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

        # Largest value (least negative) should get highest probability
        self.assertEqual(torch.argmax(result).item(), 2)  # Index of -998.0

    def test_softmax_large_positive_values(self):
        """Test softmax with large positive values

        It should handle numerical stability correctly.
        """
        operation = SoftmaxOperation()

        x = torch.tensor([1000.0, 999.0, 998.0])
        result = operation.forward(x=x)

        # Should still sum to 1 and be valid probabilities
        self.assertAlmostEqual(result.sum().item(), 1.0, places=6)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

        # Largest value should get highest probability
        self.assertEqual(torch.argmax(result).item(), 0)  # Index of 1000.0

    def test_softmax_zero_input(self):
        """Test softmax with zero input

        It should produce uniform probabilities.
        """
        operation = SoftmaxOperation()

        x = torch.zeros(4)
        result = operation.forward(x=x)

        # All should be equal (1/4 = 0.25)
        expected_prob = 1.0 / 4.0
        self.assertTrue(torch.allclose(result, torch.full_like(result, expected_prob)))

    def test_softmax_attention_pattern(self):
        """Test softmax in attention pattern

        This simulates the softmax in scaled dot-product attention.
        """
        operation = SoftmaxOperation()

        # Typical attention scores shape: [batch, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len = 2, 4, 8
        attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        result = operation.forward(x=attention_scores)

        # Each position's attention weights should sum to 1
        # Sum along the last dimension (seq_len)
        attention_sums = result.sum(dim=-1)  # [batch, num_heads, seq_len]
        expected_sums = torch.ones_like(attention_sums)
        self.assertTrue(torch.allclose(attention_sums, expected_sums))

        # All values should be between 0 and 1
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

    def test_softmax_temperature_effect(self):
        """Test softmax with temperature scaling

        Higher temperature should make distribution more uniform.
        """
        operation_normal = SoftmaxOperation()

        # Simulate temperature scaling by pre-scaling the input
        x = torch.tensor([1.0, 2.0, 3.0])

        # Normal softmax
        result_normal = operation_normal.forward(x=x)

        # High temperature (divide by 2)
        result_high_temp = operation_normal.forward(x=x / 2.0)

        # For high temperature, distribution should be more uniform
        # Check that the highest probability is lower
        max_prob_normal = torch.max(result_normal)
        max_prob_high_temp = torch.max(result_high_temp)

        self.assertLess(max_prob_high_temp.item(), max_prob_normal.item())

    def test_softmax_gradients(self):
        """Test that gradients flow correctly through softmax

        Softmax should be differentiable.
        """
        operation = SoftmaxOperation()

        x = torch.randn(5, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

    def test_softmax_different_dtypes(self):
        """Test softmax with different tensor dtypes

        It should work with various floating point precisions.
        """
        operation = SoftmaxOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)
                # Check that it sums to 1 (allowing for precision differences)
                if dtype == torch.float16:
                    self.assertAlmostEqual(result.sum().item(), 1.0, places=3)
                else:
                    self.assertAlmostEqual(result.sum().item(), 1.0, places=6)

    def test_softmax_preserves_shape(self):
        """Test that softmax preserves tensor shape

        Output should have the same shape as input.
        """
        operation = SoftmaxOperation()

        shapes = [(10,), (5, 8), (2, 3, 4), (2, 3, 4, 5)]

        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                result = operation.forward(x=x)
                self.assertEqual(result.shape, x.shape)

    def test_softmax_parameter_access(self):
        """Test that softmax dimension parameter is accessible

        The dim parameter should be stored as an instance attribute.
        """
        operation = SoftmaxOperation(dim=1)
        self.assertEqual(operation.dim, 1)

        operation_default = SoftmaxOperation()
        self.assertEqual(operation_default.dim, -1)  # Default is last dimension

    def test_softmax_batch_consistency(self):
        """Test that softmax works consistently across batch dimensions

        Each batch element should be processed independently.
        """
        operation = SoftmaxOperation()

        batch_size, feature_dim = 3, 4
        x = torch.randn(batch_size, feature_dim)

        result = operation.forward(x=x)

        # Each batch element should sum to 1
        for batch_idx in range(batch_size):
            batch_sum = result[batch_idx].sum()
            self.assertAlmostEqual(batch_sum.item(), 1.0, places=6)

    def test_softmax_numerical_stability(self):
        """Test numerical stability of softmax

        It should handle various input ranges without overflow/underflow.
        """
        operation = SoftmaxOperation()

        # Test with very different scales
        test_inputs = [
            torch.randn(10) * 0.01,    # Small values
            torch.randn(10) * 100,     # Large values
            torch.randn(10) * 0.01 + 1000,  # Large offset
            torch.randn(10) * 100 - 500,    # Negative large offset
        ]

        for x in test_inputs:
            result = operation.forward(x=x)

            # Should always produce valid probabilities
            self.assertTrue(torch.all(result >= 0))
            self.assertTrue(torch.all(result <= 1))
            self.assertAlmostEqual(result.sum().item(), 1.0, places=5)
            self.assertTrue(torch.all(torch.isfinite(result)))