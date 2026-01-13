"""Test the Add operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.add import AddOperation


class TestAddOperation(unittest.TestCase):
    """Test the Add operation"""

    def test_basic_addition(self):
        """Test basic element-wise addition of two tensors

        It should correctly add corresponding elements.
        """
        operation = AddOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([[1.5, 3.5], [5.5, 7.5]])

        self.assertTrue(torch.allclose(result, expected))

    def test_add_different_shapes_broadcasting(self):
        """Test addition with broadcasting

        It should handle broadcasting when tensor shapes are compatible.
        """
        operation = AddOperation()

        # Test adding vector to matrix (broadcasting)
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        b = torch.tensor([0.5, 1.5])  # [2] -> broadcasted to [2, 2]

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([[1.5, 3.5], [3.5, 5.5]])

        self.assertTrue(torch.allclose(result, expected))

    def test_add_same_tensor(self):
        """Test adding a tensor to itself

        It should double each element.
        """
        operation = AddOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = operation.forward(a=a, b=a)
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_add_zeros(self):
        """Test adding zero tensor

        It should return the original tensor unchanged.
        """
        operation = AddOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        zeros = torch.zeros_like(a)

        result = operation.forward(a=a, b=zeros)
        self.assertTrue(torch.allclose(result, a))

    def test_add_negative_numbers(self):
        """Test addition with negative numbers

        It should handle negative values correctly.
        """
        operation = AddOperation()

        a = torch.tensor([1.0, -2.0, 3.0])
        b = torch.tensor([-1.0, 2.0, -3.0])

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([0.0, 0.0, 0.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_add_different_dtypes(self):
        """Test addition with different tensor dtypes

        It should preserve dtype appropriately.
        """
        operation = AddOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                a = torch.tensor([1.0, 2.0], dtype=dtype)
                b = torch.tensor([0.5, 1.5], dtype=dtype)

                result = operation.forward(a=a, b=b)

                self.assertEqual(result.dtype, dtype)
                expected = torch.tensor([1.5, 3.5], dtype=dtype)
                self.assertTrue(torch.allclose(result, expected))

    def test_add_gradients(self):
        """Test that gradients flow correctly through addition

        Both input tensors should receive gradients.
        """
        operation = AddOperation()

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([0.5, 1.5], requires_grad=True)

        result = operation.forward(a=a, b=b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

        # Both should get gradient of 1.0
        self.assertTrue(torch.allclose(a.grad, torch.ones_like(a)))
        self.assertTrue(torch.allclose(b.grad, torch.ones_like(b)))

    def test_add_large_tensors(self):
        """Test addition with large tensors

        It should handle tensors of various sizes efficiently.
        """
        operation = AddOperation()

        # Test with reasonably large tensors
        a = torch.randn(32, 64, 128)
        b = torch.randn(32, 64, 128)

        result = operation.forward(a=a, b=b)
        expected = a + b

        self.assertTrue(torch.allclose(result, expected))

    def test_add_scalar_broadcasting(self):
        """Test addition with scalar broadcasting

        It should work when one tensor is a scalar.
        """
        operation = AddOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scalar = torch.tensor(1.0)

        result = operation.forward(a=a, b=scalar)
        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_add_commutative(self):
        """Test that addition is commutative

        a + b should equal b + a.
        """
        operation = AddOperation()

        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        result_ab = operation.forward(a=a, b=b)
        result_ba = operation.forward(a=b, b=a)

        self.assertTrue(torch.allclose(result_ab, result_ba))

    def test_add_associative(self):
        """Test that addition is associative

        (a + b) + c should equal a + (b + c).
        """
        operation = AddOperation()

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)

        result_left = operation.forward(a=operation.forward(a=a, b=b), b=c)
        result_right = operation.forward(a=a, b=operation.forward(a=b, b=c))

        self.assertTrue(torch.allclose(result_left, result_right))

    def test_add_identity_element(self):
        """Test that zero is the identity element

        a + 0 should equal a.
        """
        operation = AddOperation()

        a = torch.randn(5, 6)
        zero = torch.zeros_like(a)

        result = operation.forward(a=a, b=zero)
        self.assertTrue(torch.allclose(result, a))