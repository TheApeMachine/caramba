"""Test the Clamp operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.clamp import ClampOperation


class TestClampOperation(unittest.TestCase):
    """Test the Clamp operation"""

    def test_clamp_both_bounds(self):
        """Test clamping with both minimum and maximum bounds

        It should constrain values within the specified range.
        """
        operation = ClampOperation(min_val=-1.0, max_val=1.0)

        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = operation.forward(x=x)
        expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_min_only(self):
        """Test clamping with minimum bound only

        It should constrain only the lower bound.
        """
        operation = ClampOperation(min_val=0.0)

        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        result = operation.forward(x=x)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 2.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_max_only(self):
        """Test clamping with maximum bound only

        It should constrain only the upper bound.
        """
        operation = ClampOperation(max_val=1.0)

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 1.5, 2.0])
        result = operation.forward(x=x)
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 1.0, 1.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_no_bounds(self):
        """Test clamping with no bounds specified

        It should return the input unchanged.
        """
        operation = ClampOperation()

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = operation.forward(x=x)

        self.assertTrue(torch.allclose(result, x))

    def test_clamp_multidimensional(self):
        """Test clamping with multidimensional tensors

        It should apply clamping element-wise to all dimensions.
        """
        operation = ClampOperation(min_val=-0.5, max_val=0.5)

        x = torch.tensor([[[-1.0, 0.0, 1.0]], [[-0.2, 0.2, 0.8]]])
        result = operation.forward(x=x)
        expected = torch.tensor([[[-0.5, 0.0, 0.5]], [[-0.2, 0.2, 0.5]]])

        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_different_dtypes(self):
        """Test clamping with different tensor dtypes

        It should work with various floating point precisions.
        """
        operation = ClampOperation(min_val=-1.0, max_val=1.0)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)
                expected = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=dtype)
                self.assertTrue(torch.allclose(result, expected))

    def test_clamp_gradients(self):
        """Test that gradients flow correctly through clamping

        Gradients should be preserved for values within bounds and zeroed outside bounds.
        """
        operation = ClampOperation(min_val=-1.0, max_val=1.0)

        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None

        # Gradients should be 1.0 for values within bounds, 0.0 for clamped values
        expected_grad = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_clamp_all_clamped(self):
        """Test clamping when all values are outside bounds

        All values should be clamped to the appropriate bound.
        """
        operation = ClampOperation(min_val=0.0, max_val=1.0)

        x = torch.tensor([-1.0, -0.5, 1.5, 2.0])
        result = operation.forward(x=x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_identity_range(self):
        """Test clamping with infinite bounds

        It should act as identity function.
        """
        operation = ClampOperation(min_val=float('-inf'), max_val=float('inf'))

        x = torch.randn(5, 3)
        result = operation.forward(x=x)

        self.assertTrue(torch.allclose(result, x))

    def test_clamp_preserves_shape(self):
        """Test that clamping preserves tensor shape

        Output should have the same shape as input.
        """
        operation = ClampOperation(min_val=-0.5, max_val=0.5)

        shapes = [(10,), (5, 8), (2, 3, 4), (2, 3, 4, 5)]

        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                result = operation.forward(x=x)
                self.assertEqual(result.shape, x.shape)

    def test_clamp_parameter_validation(self):
        """Test that clamp parameters are stored correctly

        The min_val and max_val should be accessible as instance attributes.
        """
        operation = ClampOperation(min_val=-2.0, max_val=3.0)
        self.assertEqual(operation.min_val, -2.0)
        self.assertEqual(operation.max_val, 3.0)

        operation_default = ClampOperation()
        self.assertIsNone(operation_default.min_val)
        self.assertIsNone(operation_default.max_val)