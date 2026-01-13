"""Test the View As Heads operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.view_as_heads import ViewAsHeadsOperation


class TestViewAsHeadsOperation(unittest.TestCase):
    """Test the View As Heads operation"""

    def test_basic_view_as_heads(self):
        """Test basic view_as_heads functionality

        It should reshape [batch, seq_len, hidden_dim] to [batch, num_heads, seq_len, head_dim].
        """
        num_heads = 4
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        batch_size, seq_len, hidden_dim = 2, 8, 64  # hidden_dim must be divisible by num_heads

        x = torch.randn(batch_size, seq_len, hidden_dim)
        result = operation.forward(x=x)

        # Expected shape: [batch, num_heads, seq_len, head_dim] (after transpose)
        head_dim = hidden_dim // num_heads
        expected_shape = (batch_size, num_heads, seq_len, head_dim)

        self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_single_head(self):
        """Test view_as_heads with single head

        Should work with num_heads=1.
        """
        num_heads = 1
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        batch_size, seq_len, hidden_dim = 1, 4, 32

        x = torch.randn(batch_size, seq_len, hidden_dim)
        result = operation.forward(x=x)

        # Should transpose to [batch, num_heads, seq_len, head_dim]
        expected_shape = (batch_size, num_heads, seq_len, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_transpose(self):
        """Test that view_as_heads performs the transpose correctly

        The operation should transpose seq_len and num_heads dimensions.
        """
        num_heads = 2
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        batch_size, seq_len, hidden_dim = 1, 3, 8
        head_dim = hidden_dim // num_heads  # 4

        x = torch.randn(batch_size, seq_len, hidden_dim)
        result = operation.forward(x=x)

        # The result should be [batch, num_heads, seq_len, head_dim]
        # because of the transpose(1, 2) in the implementation
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        self.assertEqual(result.shape, expected_shape)

        # Verify the transpose happened correctly
        # Original view would be [batch, seq_len, num_heads, head_dim]
        # After transpose(1, 2): [batch, num_heads, seq_len, head_dim]
        manual_view = x.view(batch_size, seq_len, num_heads, head_dim)
        manual_transpose = manual_view.transpose(1, 2)

        self.assertTrue(torch.allclose(result, manual_transpose))

    def test_view_as_heads_different_head_counts(self):
        """Test view_as_heads with different numbers of heads

        Should work with various head counts that divide the hidden dimension.
        """
        batch_size, seq_len, hidden_dim = 1, 4, 32

        for num_heads in [2, 4, 8, 16]:
            with self.subTest(num_heads=num_heads):
                if hidden_dim % num_heads == 0:  # Only test valid divisions
                    operation = ViewAsHeadsOperation(num_heads=num_heads)

                    x = torch.randn(batch_size, seq_len, hidden_dim)
                    result = operation.forward(x=x)

                    head_dim = hidden_dim // num_heads
                    expected_shape = (batch_size, num_heads, seq_len, head_dim)
                    self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_invalid_head_count(self):
        """Test that incompatible dimensions raise ValueError

        hidden_dim must be divisible by num_heads.
        """
        operation = ViewAsHeadsOperation(num_heads=3)
        x = torch.randn(1, 2, 8)  # hidden_dim=8, not divisible by 3

        with self.assertRaises(ValueError) as context:
            operation.forward(x=x)

        self.assertIn("must be divisible by", str(context.exception))

    def test_view_as_heads_zero_heads(self):
        """Test that zero heads raises ValueError

        num_heads must be > 0.
        """
        with self.assertRaises(ValueError) as context:
            ViewAsHeadsOperation(num_heads=0)

        self.assertIn("num_heads must be > 0", str(context.exception))

    def test_view_as_heads_negative_heads(self):
        """Test that negative heads raises ValueError

        num_heads must be > 0.
        """
        with self.assertRaises(ValueError) as context:
            ViewAsHeadsOperation(num_heads=-1)

        self.assertIn("num_heads must be > 0", str(context.exception))

    def test_view_as_heads_gradients(self):
        """Test that gradients flow correctly through view_as_heads

        The operation should preserve gradient flow.
        """
        num_heads = 2
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        x = torch.randn(1, 3, 8, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_view_as_heads_memory_contiguity(self):
        """Test that view_as_heads maintains memory contiguity when possible

        The operation should not unnecessarily copy data.
        """
        num_heads = 2
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        x = torch.randn(2, 4, 8)
        result = operation.forward(x=x)

        # The result should share memory with the input (it's a view)
        # Check that they share the same data pointer
        self.assertEqual(result.data_ptr(), x.data_ptr())

    def test_view_as_heads_different_batch_sizes(self):
        """Test view_as_heads with different batch sizes

        Should handle various batch configurations.
        """
        num_heads = 4

        for batch_size in [1, 3, 8]:
            with self.subTest(batch_size=batch_size):
                operation = ViewAsHeadsOperation(num_heads=num_heads)

                seq_len, hidden_dim = 6, 32
                x = torch.randn(batch_size, seq_len, hidden_dim)
                result = operation.forward(x=x)

                head_dim = hidden_dim // num_heads
                expected_shape = (batch_size, num_heads, seq_len, head_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_different_sequence_lengths(self):
        """Test view_as_heads with different sequence lengths

        Should handle various sequence lengths.
        """
        num_heads = 2

        for seq_len in [1, 4, 16, 32]:
            with self.subTest(seq_len=seq_len):
                operation = ViewAsHeadsOperation(num_heads=num_heads)

                batch_size, hidden_dim = 1, 16
                x = torch.randn(batch_size, seq_len, hidden_dim)
                result = operation.forward(x=x)

                head_dim = hidden_dim // num_heads
                expected_shape = (batch_size, num_heads, seq_len, head_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_different_hidden_dims(self):
        """Test view_as_heads with different hidden dimensions

        Should work as long as hidden_dim is divisible by num_heads.
        """
        num_heads = 4

        for hidden_dim in [16, 32, 64, 128]:
            with self.subTest(hidden_dim=hidden_dim):
                operation = ViewAsHeadsOperation(num_heads=num_heads)

                batch_size, seq_len = 1, 2
                x = torch.randn(batch_size, seq_len, hidden_dim)
                result = operation.forward(x=x)

                head_dim = hidden_dim // num_heads
                expected_shape = (batch_size, num_heads, seq_len, head_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_view_as_heads_preserves_dtype(self):
        """Test that view_as_heads preserves tensor dtype

        Output should have the same dtype as input.
        """
        num_heads = 2
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 2, 8, dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)

    def test_view_as_heads_attention_compatibility(self):
        """Test that view_as_heads output is compatible with attention operations

        The output shape should be correct for attention mechanisms.
        """
        num_heads = 8
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        # Typical transformer dimensions
        batch_size, seq_len, hidden_dim = 4, 32, 512

        x = torch.randn(batch_size, seq_len, hidden_dim)
        result = operation.forward(x=x)

        # Should produce [batch, num_heads, seq_len, head_dim]
        head_dim = hidden_dim // num_heads
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        self.assertEqual(result.shape, expected_shape)

        # Verify the dimensions are correct for attention
        self.assertEqual(result.size(1), num_heads)  # num_heads dimension
        self.assertEqual(result.size(2), seq_len)   # seq_len dimension
        self.assertEqual(result.size(3), head_dim)  # head_dim dimension

    def test_view_as_heads_inverse_operation(self):
        """Test conceptual inverse of view_as_heads

        While we don't have the inverse operation, we can verify the reshape logic.
        """
        num_heads = 3
        operation = ViewAsHeadsOperation(num_heads=num_heads)

        batch_size, seq_len, hidden_dim = 2, 4, 24  # 24 / 3 = 8

        x = torch.randn(batch_size, seq_len, hidden_dim)
        result = operation.forward(x=x)

        # Manually verify the reshape and transpose
        head_dim = hidden_dim // num_heads
        manual_reshape = x.view(batch_size, seq_len, num_heads, head_dim)
        manual_result = manual_reshape.transpose(1, 2)

        self.assertTrue(torch.allclose(result, manual_result))

        # Verify that transposing back gives us the reshape
        back_transpose = result.transpose(1, 2)
        self.assertTrue(torch.allclose(back_transpose, manual_reshape))