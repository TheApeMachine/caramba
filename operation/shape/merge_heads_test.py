"""Test the Merge Heads operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.merge_heads import MergeHeadsOperation


class TestMergeHeadsOperation(unittest.TestCase):
    """Test the Merge Heads operation"""

    def test_basic_merge_heads(self):
        """Test basic merge heads functionality

        It should reshape [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads * head_dim].
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
        hidden_dim = num_heads * head_dim  # 64

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        result = operation.forward(x=x)

        # Expected shape: [batch, seq_len, hidden_dim]
        expected_shape = (batch_size, seq_len, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_single_head(self):
        """Test merge heads with single head

        Should work with num_heads=1.
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 32

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        result = operation.forward(x=x)

        # Should collapse the head dimension
        expected_shape = (batch_size, seq_len, head_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_transpose_and_view(self):
        """Test that merge_heads performs transpose and view correctly

        The operation should transpose then reshape to combine heads.
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 3, 4

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        result = operation.forward(x=x)

        # Manual transpose and view
        manual_transpose = x.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        manual_result = manual_transpose.contiguous().view(batch_size, seq_len, -1)

        self.assertTrue(torch.allclose(result, manual_result))

    def test_merge_heads_different_head_counts(self):
        """Test merge heads with different numbers of heads

        Should work with various head counts.
        """
        operation = MergeHeadsOperation()

        batch_size, seq_len, head_dim = 1, 4, 8

        for num_heads in [1, 2, 4, 8]:
            with self.subTest(num_heads=num_heads):
                x = torch.randn(batch_size, num_heads, seq_len, head_dim)
                result = operation.forward(x=x)

                hidden_dim = num_heads * head_dim
                expected_shape = (batch_size, seq_len, hidden_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_round_trip_with_view_as_heads(self):
        """Test round-trip compatibility with view_as_heads

        Merging heads after viewing as heads should restore original shape.
        """
        from caramba.operation.shape.view_as_heads import ViewAsHeadsOperation

        num_heads = 4
        view_op = ViewAsHeadsOperation(num_heads=num_heads)
        merge_op = MergeHeadsOperation()

        batch_size, seq_len, hidden_dim = 2, 6, 32

        # Original tensor
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # View as heads then merge back
        heads = view_op.forward(x=x)
        result = merge_op.forward(x=heads)

        # Should get back the original shape
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.allclose(result, x))

    def test_merge_heads_gradients(self):
        """Test that gradients flow correctly through merge_heads

        The operation should preserve gradient flow.
        """
        operation = MergeHeadsOperation()

        x = torch.randn(1, 3, 4, 8, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_merge_heads_memory_contiguity(self):
        """Test that merge_heads makes the result contiguous

        The operation calls .contiguous() so result should be contiguous.
        """
        operation = MergeHeadsOperation()

        x = torch.randn(2, 4, 6, 8)
        result = operation.forward(x=x)

        # Result should be contiguous after the .contiguous() call
        self.assertTrue(result.is_contiguous())

    def test_merge_heads_different_batch_sizes(self):
        """Test merge heads with different batch sizes

        Should handle various batch configurations.
        """
        operation = MergeHeadsOperation()

        num_heads, seq_len, head_dim = 2, 4, 16

        for batch_size in [1, 3, 5]:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, num_heads, seq_len, head_dim)
                result = operation.forward(x=x)

                hidden_dim = num_heads * head_dim
                expected_shape = (batch_size, seq_len, hidden_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_different_sequence_lengths(self):
        """Test merge heads with different sequence lengths

        Should handle various sequence lengths.
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, head_dim = 1, 2, 8

        for seq_len in [1, 4, 16, 32]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(batch_size, num_heads, seq_len, head_dim)
                result = operation.forward(x=x)

                hidden_dim = num_heads * head_dim
                expected_shape = (batch_size, seq_len, hidden_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_different_head_dims(self):
        """Test merge heads with different head dimensions

        Should work with various head dimensions.
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, seq_len = 1, 3, 2

        for head_dim in [4, 8, 16, 32]:
            with self.subTest(head_dim=head_dim):
                x = torch.randn(batch_size, num_heads, seq_len, head_dim)
                result = operation.forward(x=x)

                hidden_dim = num_heads * head_dim
                expected_shape = (batch_size, seq_len, hidden_dim)
                self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_preserves_dtype(self):
        """Test that merge_heads preserves tensor dtype

        Output should have the same dtype as input.
        """
        operation = MergeHeadsOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 2, 3, 4, dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)

    def test_merge_heads_attention_compatibility(self):
        """Test that merge_heads output is compatible with transformer layers

        The output shape should be correct for subsequent transformer layers.
        """
        operation = MergeHeadsOperation()

        # Typical attention output dimensions
        batch_size, num_heads, seq_len, head_dim = 4, 8, 32, 64

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        result = operation.forward(x=x)

        # Should produce [batch, seq_len, hidden_dim]
        hidden_dim = num_heads * head_dim
        expected_shape = (batch_size, seq_len, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

        # Verify the dimensions are correct for transformer layers
        self.assertEqual(result.size(0), batch_size)  # batch dimension
        self.assertEqual(result.size(1), seq_len)    # seq_len dimension
        self.assertEqual(result.size(2), hidden_dim) # hidden_dim dimension

    def test_merge_heads_single_element(self):
        """Test merge heads with minimal dimensions

        Should work with 1x1x1x1 tensors.
        """
        operation = MergeHeadsOperation()

        x = torch.randn(1, 1, 1, 1)
        result = operation.forward(x=x)

        expected_shape = (1, 1, 1)
        self.assertEqual(result.shape, expected_shape)

    def test_merge_heads_preserves_data_integrity(self):
        """Test that merge_heads preserves data correctly

        All data should be preserved through the transpose and reshape.
        """
        operation = MergeHeadsOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 3, 4, 2

        # Create a predictable pattern
        x = torch.arange(batch_size * num_heads * seq_len * head_dim, dtype=torch.float32)
        x = x.view(batch_size, num_heads, seq_len, head_dim)

        result = operation.forward(x=x)

        # Flatten and check that all values are preserved
        result_flat = result.view(-1)
        x_flat = x.transpose(1, 2).contiguous().view(-1)

        self.assertTrue(torch.allclose(result_flat, x_flat))