"""Test the Transpose operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.transpose import TransposeOperation


class TestTransposeOperation(unittest.TestCase):
    """Test the Transpose operation"""

    def test_basic_transpose(self):
        """Test basic transpose functionality

        It should swap the specified dimensions.
        """
        operation = TransposeOperation(dim0=0, dim1=1)

        x = torch.randn(3, 4, 5)
        result = operation.forward(x=x)

        # Should swap dimensions 0 and 1
        expected_shape = (4, 3, 5)
        self.assertEqual(result.shape, expected_shape)

        # Verify the transpose is correct
        expected = x.transpose(0, 1)
        self.assertTrue(torch.allclose(result, expected))

    def test_transpose_different_dimensions(self):
        """Test transpose with different dimension pairs

        Should work with various dimension pairs.
        """
        x = torch.randn(2, 3, 4, 5)

        test_cases = [
            (0, 1),  # First and second
            (0, 2),  # First and third
            (1, 2),  # Second and third
            (0, 3),  # First and last
            (1, 3),  # Second and last
            (2, 3),  # Third and last
        ]

        for dim0, dim1 in test_cases:
            with self.subTest(dim0=dim0, dim1=dim1):
                operation = TransposeOperation(dim0=dim0, dim1=dim1)
                result = operation.forward(x=x)

                expected = x.transpose(dim0, dim1)
                self.assertTrue(torch.allclose(result, expected))

    def test_transpose_same_dimension(self):
        """Test transpose with same dimension

        Transposing a dimension with itself should be identity.
        """
        operation = TransposeOperation(dim0=1, dim1=1)

        x = torch.randn(3, 4, 5)
        result = operation.forward(x=x)

        # Should be unchanged
        self.assertTrue(torch.allclose(result, x))

    def test_transpose_gradients(self):
        """Test that gradients flow correctly through transpose

        Transpose should preserve gradient flow.
        """
        operation = TransposeOperation(dim0=0, dim1=2)

        x = torch.randn(2, 3, 4, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_transpose_memory_efficiency(self):
        """Test that transpose creates views when possible

        Transpose should not copy data unnecessarily.
        """
        operation = TransposeOperation(dim0=0, dim1=1)

        x = torch.randn(4, 5)
        result = operation.forward(x=x)

        # For 2D tensors, transpose should return a view
        # Check if they share memory (they should for simple transposes)
        self.assertEqual(result.data_ptr(), x.data_ptr())

    def test_transpose_multidimensional(self):
        """Test transpose with higher dimensional tensors

        Should work with tensors of various shapes.
        """
        shapes = [(2, 3, 4), (3, 4, 5, 6), (2, 2, 2, 2, 2)]

        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                operation = TransposeOperation(dim0=0, dim1=len(shape)-1)
                result = operation.forward(x=x)

                expected = x.transpose(0, len(shape)-1)
                self.assertTrue(torch.allclose(result, expected))

    def test_transpose_preserves_dtype(self):
        """Test that transpose preserves tensor dtype

        Output should have the same dtype as input.
        """
        operation = TransposeOperation(dim0=1, dim1=2)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 3, 4, dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)

    def test_transpose_attention_patterns(self):
        """Test transpose patterns common in attention mechanisms

        These are typical transposes used in transformers.
        """
        # Common attention transpose patterns
        patterns = [
            # Q*K^T: transpose last two dimensions
            ((2, 4, 8, 16), 2, 3),  # [batch, heads, seq, dim] -> transpose seq and dim
            # Various attention-related transposes
            ((2, 8, 32, 64), 1, 2),  # heads and seq
            ((4, 16, 128), 0, 2),    # batch and dim
        ]

        for shape, dim0, dim1 in patterns:
            with self.subTest(shape=shape, dim0=dim0, dim1=dim1):
                x = torch.randn(shape)
                operation = TransposeOperation(dim0=dim0, dim1=dim1)
                result = operation.forward(x=x)

                expected = x.transpose(dim0, dim1)
                self.assertTrue(torch.allclose(result, expected))

    def test_transpose_negative_dimensions(self):
        """Test transpose with negative dimension indices

        Should work with negative indices like PyTorch transpose.
        """
        x = torch.randn(2, 3, 4, 5)

        # Test negative indices
        operation = TransposeOperation(dim0=-1, dim1=-2)  # Last two dimensions
        result = operation.forward(x=x)

        expected = x.transpose(-1, -2)
        self.assertTrue(torch.allclose(result, expected))

    def test_transpose_round_trip(self):
        """Test that transposing twice returns to original

        transpose(dim0, dim1) followed by transpose(dim0, dim1) should be identity.
        """
        operation = TransposeOperation(dim0=1, dim1=2)

        x = torch.randn(3, 4, 5, 6)

        # Transpose twice
        result1 = operation.forward(x=x)
        result2 = operation.forward(x=result1)

        # Should be back to original
        self.assertTrue(torch.allclose(result2, x))

    def test_transpose_commutativity(self):
        """Test that transpose(dim0, dim1) == transpose(dim1, dim0)

        Transpose should be symmetric.
        """
        x = torch.randn(3, 4, 5)

        op1 = TransposeOperation(dim0=0, dim1=2)
        op2 = TransposeOperation(dim0=2, dim1=0)

        result1 = op1.forward(x=x)
        result2 = op2.forward(x=x)

        # Should be identical
        self.assertTrue(torch.allclose(result1, result2))

    def test_transpose_parameter_validation(self):
        """Test that transpose parameters are stored correctly

        The dim0 and dim1 should be accessible as instance attributes.
        """
        operation = TransposeOperation(dim0=1, dim1=3)
        self.assertEqual(operation.dim0, 1)
        self.assertEqual(operation.dim1, 3)

    def test_transpose_identity_operations(self):
        """Test transposes that result in no change

        Certain transposes should be identity operations.
        """
        x = torch.randn(3, 4, 5)

        # Transpose with same dimension
        operation = TransposeOperation(dim0=1, dim1=1)
        result = operation.forward(x=x)
        self.assertTrue(torch.allclose(result, x))

    def test_transpose_with_attention_workflow(self):
        """Test transpose in a typical attention workflow

        Simulate the transposes used in multi-head attention.
        """
        # Simulate attention workflow transposes
        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 32

        # Start with Q, K, V in [batch, seq_len, hidden_dim]
        hidden_dim = num_heads * head_dim
        q = torch.randn(batch_size, seq_len, hidden_dim)
        k = torch.randn(batch_size, seq_len, hidden_dim)
        v = torch.randn(batch_size, seq_len, hidden_dim)

        # Simulate view_as_heads (conceptually)
        # After viewing: [batch, num_heads, seq_len, head_dim]

        # For attention, we need [batch, num_heads, seq_len, seq_len] for scores
        # This would involve transposes in the actual attention computation

        # Test the transpose operations that would be used
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Transpose for different attention patterns
        transpose_ops = [
            TransposeOperation(2, 3),  # seq_len dimensions
            TransposeOperation(1, 2),  # heads and seq_len
        ]

        for op in transpose_ops:
            result = op.forward(x=scores)
            # Just verify it runs without error and shape is correct
            self.assertEqual(len(result.shape), 4)

    def test_transpose_edge_cases(self):
        """Test transpose with edge case tensor shapes

        Should handle unusual but valid tensor shapes.
        """
        # Test with minimal dimensions
        x = torch.randn(1, 1, 1)
        operation = TransposeOperation(dim0=0, dim1=2)
        result = operation.forward(x=x)

        expected = x.transpose(0, 2)
        self.assertTrue(torch.allclose(result, expected))

        # Test with large number of dimensions
        x = torch.randn(2, 2, 2, 2, 2, 2)
        operation = TransposeOperation(dim0=2, dim1=4)
        result = operation.forward(x=x)

        expected = x.transpose(2, 4)
        self.assertTrue(torch.allclose(result, expected))