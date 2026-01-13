"""Test the Mul operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.mul import MulOperation


class TestMulOperation(unittest.TestCase):
    """Test the Mul operation"""

    def test_basic_elementwise_multiplication(self):
        """Test basic element-wise multiplication

        It should multiply corresponding elements.
        """
        operation = MulOperation()

        a = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([[2.0, 6.0], [12.0, 20.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_multiplication_with_ones(self):
        """Test multiplication with tensor of ones

        It should return the original tensor.
        """
        operation = MulOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ones = torch.ones_like(a)

        result = operation.forward(a=a, b=ones)
        self.assertTrue(torch.allclose(result, a))

    def test_multiplication_with_zeros(self):
        """Test multiplication with tensor of zeros

        It should return zeros.
        """
        operation = MulOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        zeros = torch.zeros_like(a)

        result = operation.forward(a=a, b=zeros)
        self.assertTrue(torch.allclose(result, zeros))

    def test_broadcasting_multiplication(self):
        """Test element-wise multiplication with broadcasting

        It should handle broadcasting when tensor shapes are compatible.
        """
        operation = MulOperation()

        # Matrix-vector multiplication (broadcasting)
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        b = torch.tensor([2.0, 3.0])  # [2] -> broadcasted to [2, 2]

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([[2.0, 6.0], [6.0, 12.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_negative_number_multiplication(self):
        """Test multiplication with negative numbers

        It should handle negative values correctly.
        """
        operation = MulOperation()

        a = torch.tensor([1.0, -2.0, 3.0])
        b = torch.tensor([-1.0, 2.0, -3.0])

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([-1.0, -4.0, -9.0])

        self.assertTrue(torch.allclose(result, expected))

    def test_different_dtypes(self):
        """Test multiplication with different tensor dtypes

        It should preserve dtype appropriately.
        """
        operation = MulOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                a = torch.tensor([2.0, 3.0], dtype=dtype)
                b = torch.tensor([4.0, 5.0], dtype=dtype)

                result = operation.forward(a=a, b=b)

                self.assertEqual(result.dtype, dtype)
                expected = torch.tensor([8.0, 15.0], dtype=dtype)
                self.assertTrue(torch.allclose(result, expected))

    def test_gradients(self):
        """Test that gradients flow correctly through multiplication

        Both input tensors should receive gradients.
        """
        operation = MulOperation()

        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0], requires_grad=True)

        result = operation.forward(a=a, b=b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

        # d/da(a*b) = b, d/db(a*b) = a
        self.assertTrue(torch.allclose(a.grad, b))
        self.assertTrue(torch.allclose(b.grad, a))

    def test_attention_weight_application(self):
        """Test applying attention weights (common use case)

        This simulates multiplying attention weights by values.
        """
        operation = MulOperation()

        # Attention weights [batch, num_heads, seq_len, seq_len]
        weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)

        # Values [batch, num_heads, seq_len, head_dim]
        values = torch.randn(2, 4, 8, 32)

        # This wouldn't work directly - need to expand weights
        # Let's simulate a simpler case
        weights_simple = torch.tensor([[0.5, 0.3, 0.2]])  # [1, 3]
        values_simple = torch.tensor([[1.0, 2.0, 3.0]])    # [1, 3]

        result = operation.forward(a=weights_simple, b=values_simple)
        expected = torch.tensor([[0.5, 0.6, 0.6]])

        self.assertTrue(torch.allclose(result, expected))

    def test_commutative(self):
        """Test that multiplication is commutative

        a * b should equal b * a.
        """
        operation = MulOperation()

        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        result_ab = operation.forward(a=a, b=b)
        result_ba = operation.forward(a=b, b=a)

        self.assertTrue(torch.allclose(result_ab, result_ba))

    def test_associative(self):
        """Test that multiplication is associative

        (a * b) * c should equal a * (b * c).
        """
        operation = MulOperation()

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)

        result_left = operation.forward(a=operation.forward(a=a, b=b), b=c)
        result_right = operation.forward(a=a, b=operation.forward(a=b, b=c))

        self.assertTrue(torch.allclose(result_left, result_right))

    def test_identity_element(self):
        """Test that one is the identity element

        a * 1 should equal a.
        """
        operation = MulOperation()

        a = torch.randn(5, 6)
        ones = torch.ones_like(a)

        result = operation.forward(a=a, b=ones)
        self.assertTrue(torch.allclose(result, a))

    def test_zero_element(self):
        """Test that zero is the absorbing element

        a * 0 should equal 0.
        """
        operation = MulOperation()

        a = torch.randn(3, 4)
        zeros = torch.zeros_like(a)

        result = operation.forward(a=a, b=zeros)
        self.assertTrue(torch.allclose(result, zeros))

    def test_scalar_broadcasting(self):
        """Test multiplication with scalar broadcasting

        It should work when one tensor is a scalar.
        """
        operation = MulOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scalar = torch.tensor(3.0)

        result = operation.forward(a=a, b=scalar)
        expected = torch.tensor([[3.0, 6.0], [9.0, 12.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_large_tensors(self):
        """Test multiplication with large tensors

        It should handle tensors of various sizes efficiently.
        """
        operation = MulOperation()

        # Test with reasonably large tensors
        a = torch.randn(64, 128)
        b = torch.randn(64, 128)

        result = operation.forward(a=a, b=b)
        expected = a * b

        self.assertTrue(torch.allclose(result, expected))