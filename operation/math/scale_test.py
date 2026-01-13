"""Test the Scale operation"""
from __future__ import annotations

import math
import unittest

import torch

from caramba.operation.math.scale import ScaleOperation


class TestScaleOperation(unittest.TestCase):
    """Test the Scale operation"""

    def test_scale_by_scalar(self):
        """Test scaling by a scalar value

        It should multiply all elements by the scalar.
        """
        operation = ScaleOperation(scale=2.5)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = operation.forward(x=x)
        expected = torch.tensor([[2.5, 5.0], [7.5, 10.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_scale_by_tensor(self):
        """Test scaling by a tensor (element-wise multiplication)

        It should multiply element-wise with the tensor.
        """
        scale_tensor = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        operation = ScaleOperation(scale=scale_tensor)

        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        result = operation.forward(x=x)
        expected = scale_tensor  # x * scale_tensor with x=1

        self.assertTrue(torch.allclose(result, expected))

    def test_scale_identity(self):
        """Test scaling by identity (1.0)

        It should return the input unchanged.
        """
        operation = ScaleOperation(scale=1.0)

        x = torch.randn(3, 4)
        result = operation.forward(x=x)

        self.assertTrue(torch.allclose(result, x))

    def test_scale_zero(self):
        """Test scaling by zero

        It should return all zeros.
        """
        operation = ScaleOperation(scale=0.0)

        x = torch.randn(2, 3)
        result = operation.forward(x=x)

        self.assertTrue(torch.allclose(result, torch.zeros_like(x)))

    def test_scale_negative(self):
        """Test scaling by negative values

        It should handle negative scaling correctly.
        """
        operation = ScaleOperation(scale=-2.0)

        x = torch.tensor([1.0, -2.0, 3.0])
        result = operation.forward(x=x)
        expected = torch.tensor([-2.0, 4.0, -6.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_temperature_scaling(self):
        """Test temperature scaling (common use case)

        This simulates temperature scaling in softmax.
        """
        temperature = 0.5
        operation = ScaleOperation(scale=1.0/temperature)  # Inverse temperature

        logits = torch.tensor([1.0, 2.0, 3.0])
        result = operation.forward(x=logits)
        expected = logits * 2.0  # 1/0.5 = 2.0

        self.assertTrue(torch.allclose(result, expected))

    def test_attention_scaling(self):
        """Test attention scaling by sqrt(head_dim)

        This simulates the scaling in scaled dot-product attention.
        """
        head_dim = 64
        scale_factor = 1.0 / math.sqrt(head_dim)
        operation = ScaleOperation(scale=scale_factor)

        attention_scores = torch.randn(2, 4, 8, 8)
        result = operation.forward(x=attention_scores)

        # Should be scaled by 1/sqrt(64) = 1/8 = 0.125
        expected = attention_scores * scale_factor
        self.assertTrue(torch.allclose(result, expected))

    def test_gradients(self):
        """Test that gradients flow correctly through scaling

        Gradients should be scaled by the scale factor.
        """
        scale_value = 3.0
        operation = ScaleOperation(scale=scale_value)

        x = torch.randn(3, 4, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None

        # Gradient should be scaled by the scale factor
        self.assertTrue(torch.allclose(x.grad, torch.full_like(x.grad, scale_value)))

    def test_tensor_scale_gradients(self):
        """Test gradients when scaling by a tensor

        Gradients should flow correctly for tensor scaling.
        """
        scale_tensor = torch.tensor([2.0, 3.0, 4.0])
        operation = ScaleOperation(scale=scale_tensor)

        x = torch.ones(3, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None

        # Gradient should be the scale tensor values
        self.assertTrue(torch.allclose(x.grad, scale_tensor))

    def test_different_dtypes(self):
        """Test scaling with different tensor dtypes

        It should preserve dtype appropriately.
        """
        operation = ScaleOperation(scale=2.0)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)
                expected = torch.tensor([2.0, 4.0, 6.0], dtype=dtype)
                self.assertTrue(torch.allclose(result, expected))

    def test_preserves_shape(self):
        """Test that scaling preserves tensor shape

        Output should have the same shape as input.
        """
        operation = ScaleOperation(scale=1.5)

        shapes = [(10,), (5, 8), (2, 3, 4), (2, 3, 4, 5)]

        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                result = operation.forward(x=x)
                self.assertEqual(result.shape, x.shape)

    def test_parameter_storage(self):
        """Test that scale parameter is stored correctly

        The scale parameter should be accessible as an instance attribute.
        """
        scalar_scale = 2.5
        operation = ScaleOperation(scale=scalar_scale)
        self.assertEqual(operation.scale, scalar_scale)

        tensor_scale = torch.tensor([1.0, 2.0, 3.0])
        operation2 = ScaleOperation(scale=tensor_scale)
        self.assertTrue(torch.allclose(torch.as_tensor(operation2.scale), tensor_scale))

    def test_large_tensor_scaling(self):
        """Test scaling with large tensors

        It should handle tensors of various sizes efficiently.
        """
        operation = ScaleOperation(scale=0.01)

        # Test with reasonably large tensor
        x = torch.randn(128, 256)
        result = operation.forward(x=x)
        expected = x * 0.01

        self.assertTrue(torch.allclose(result, expected))

    def test_broadcasting_tensor_scale(self):
        """Test tensor scaling with broadcasting

        The scale tensor should broadcast to match input tensor shape.
        """
        # Scale tensor with shape [3] applied to tensor with shape [2, 3]
        scale_tensor = torch.tensor([1.0, 2.0, 3.0])  # [3]
        operation = ScaleOperation(scale=scale_tensor)

        x = torch.ones(2, 3)  # [2, 3]
        result = operation.forward(x=x)
        expected = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # Broadcasting

        self.assertTrue(torch.allclose(result, expected))