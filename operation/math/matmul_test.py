"""Test the MatMul operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.matmul import MatMulOperation


class TestMatMulOperation(unittest.TestCase):
    """Test the MatMul operation"""

    def test_basic_matrix_multiplication(self):
        """Test basic matrix multiplication

        It should correctly multiply two matrices.
        """
        operation = MatMulOperation()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # [2, 2]

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])  # Manual calculation

        self.assertTrue(torch.allclose(result, expected))

    def test_vector_matrix_multiplication(self):
        """Test multiplying vector by matrix

        It should handle vector-matrix multiplication.
        """
        operation = MatMulOperation()

        a = torch.tensor([1.0, 2.0])  # [2]
        b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])  # [2, 2]

        result = operation.forward(a=a, b=b)
        expected = torch.tensor([13.0, 16.0])  # [1, 2] -> [2]

        self.assertTrue(torch.allclose(result, expected))

    def test_batch_matrix_multiplication(self):
        """Test batch matrix multiplication

        It should handle batched matrix operations.
        """
        operation = MatMulOperation()

        a = torch.randn(3, 4, 5)  # [batch=3, m=4, k=5]
        b = torch.randn(3, 5, 6)  # [batch=3, k=5, n=6]

        result = operation.forward(a=a, b=b)
        expected = torch.matmul(a, b)

        self.assertEqual(result.shape, (3, 4, 6))
        self.assertTrue(torch.allclose(result, expected))

    def test_attention_pattern(self):
        """Test attention mechanism pattern (query-key multiplication)

        This is the most common use case in transformers.
        """
        operation = MatMulOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 32

        # Query and key matrices (transposed for attention)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, head_dim, seq_len)  # Note: transposed

        result = operation.forward(a=q, b=k)

        # Attention scores shape should be [batch, num_heads, seq_len, seq_len]
        self.assertEqual(result.shape, (batch_size, num_heads, seq_len, seq_len))

        # Verify against torch.matmul
        expected = torch.matmul(q, k)
        self.assertTrue(torch.allclose(result, expected))

    def test_broadcasting_cases(self):
        """Test matrix multiplication with broadcasting

        It should handle cases where dimensions can broadcast.
        """
        operation = MatMulOperation()

        # Case 1: Matrix-vector multiplication with broadcasting
        a = torch.randn(3, 4, 5)  # [3, 4, 5]
        b = torch.randn(5, 6)     # [5, 6] -> broadcasted to [3, 5, 6]? No, this won't work

        # Actually, matmul doesn't support arbitrary broadcasting like addition
        # Let's test a valid broadcasting case
        a = torch.randn(2, 3, 4)
        b = torch.randn(4, 5)

        result = operation.forward(a=a, b=b)
        expected = torch.matmul(a, b)

        self.assertEqual(result.shape, (2, 3, 5))
        self.assertTrue(torch.allclose(result, expected))

    def test_gradients(self):
        """Test that gradients flow correctly through matrix multiplication

        Both input tensors should receive gradients.
        """
        operation = MatMulOperation()

        a = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(4, 5, requires_grad=True)

        result = operation.forward(a=a, b=b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

        # Gradients should have correct shapes
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

    def test_different_dtypes(self):
        """Test matrix multiplication with different tensor dtypes

        It should work with various floating point precisions.
        """
        operation = MatMulOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                a = torch.randn(3, 4, dtype=dtype)
                b = torch.randn(4, 5, dtype=dtype)

                result = operation.forward(a=a, b=b)

                self.assertEqual(result.dtype, dtype)
                expected = torch.matmul(a, b)
                self.assertTrue(torch.allclose(result, expected))

    def test_identity_matrix_multiplication(self):
        """Test multiplication with identity matrix

        It should return the original matrix.
        """
        operation = MatMulOperation()

        a = torch.randn(4, 4)
        identity = torch.eye(4)

        result = operation.forward(a=a, b=identity)
        self.assertTrue(torch.allclose(result, a))

        result = operation.forward(a=identity, b=a)
        self.assertTrue(torch.allclose(result, a))

    def test_transpose_multiplication(self):
        """Test matrix multiplication with transposes

        Common pattern in attention mechanisms.
        """
        operation = MatMulOperation()

        a = torch.randn(6, 4)
        b = torch.randn(4, 5)

        # a @ b (normal multiplication)
        result1 = operation.forward(a=a, b=b)
        expected1 = torch.matmul(a, b)
        self.assertTrue(torch.allclose(result1, expected1))

        # Test transpose pattern: a.T @ something
        # Create compatible matrix: [4, 7] to multiply with a.t() [4, 6] -> need [6, 7]
        c = torch.randn(6, 7)
        result2 = operation.forward(a=a.t(), b=c)
        expected2 = torch.matmul(a.t(), c)
        self.assertTrue(torch.allclose(result2, expected2))

    def test_large_tensor_multiplication(self):
        """Test matrix multiplication with large tensors

        It should handle larger computations efficiently.
        """
        operation = MatMulOperation()

        # Test with reasonably large matrices
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)

        result = operation.forward(a=a, b=b)
        expected = torch.matmul(a, b)

        self.assertEqual(result.shape, (64, 256))
        self.assertTrue(torch.allclose(result, expected))

    def test_associativity_approximation(self):
        """Test that matrix multiplication is associative

        (A @ B) @ C should equal A @ (B @ C) (for compatible dimensions).
        """
        operation = MatMulOperation()

        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(5, 6)

        # (A @ B) @ C
        temp = operation.forward(a=a, b=b)
        result_left = operation.forward(a=temp, b=c)

        # A @ (B @ C)
        temp = operation.forward(a=b, b=c)
        result_right = operation.forward(a=a, b=temp)

        # Use higher tolerance due to floating point precision
        self.assertTrue(torch.allclose(result_left, result_right, rtol=1e-5, atol=1e-6))

    def test_distributive_approximation(self):
        """Test distributive property with scalar multiplication

        This tests that scalar multiplication distributes through matrix multiplication.
        """
        operation = MatMulOperation()

        scalar = 2.5
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)

        # scalar * (a @ b)
        result1 = scalar * operation.forward(a=a, b=b)

        # (scalar * a) @ b
        result2 = operation.forward(a=scalar * a, b=b)

        # a @ (scalar * b)
        result3 = operation.forward(a=a, b=scalar * b)

        self.assertTrue(torch.allclose(result1, result2))
        self.assertTrue(torch.allclose(result1, result3))