"""Test the Split operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.split import SplitOperation


class TestSplitOperation(unittest.TestCase):
    """Test the Split operation"""

    def test_basic_split(self):
        """Test basic tensor splitting

        It should divide tensor into equal-sized chunks.
        """
        operation = SplitOperation(split_size=2, dim=0)

        x = torch.randn(6, 4)  # 6 elements along dim 0
        result = operation.forward(x=x)

        # Should split into 3 chunks of size 2
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk.shape, (2, 4))

        # Verify the chunks are correct
        expected_chunks = x.split(2, dim=0)
        for i, chunk in enumerate(result):
            self.assertTrue(torch.allclose(chunk, expected_chunks[i]))

    def test_split_along_different_dimensions(self):
        """Test splitting along different dimensions

        Should work along any valid dimension.
        """
        x = torch.randn(3, 8, 5)

        # Split along dimension 1
        operation = SplitOperation(split_size=4, dim=1)
        result = operation.forward(x=x)

        self.assertEqual(len(result), 2)  # 8 / 4 = 2
        for chunk in result:
            self.assertEqual(chunk.shape, (3, 4, 5))

    def test_split_uneven_size(self):
        """Test splitting when dimension size is not evenly divisible

        Should create chunks of the specified size and one smaller chunk.
        """
        operation = SplitOperation(split_size=3, dim=0)

        x = torch.randn(10, 4)  # 10 elements along dim 0
        result = operation.forward(x=x)

        # Should split into chunks: [3, 3, 3, 1]
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].shape, (3, 4))
        self.assertEqual(result[1].shape, (3, 4))
        self.assertEqual(result[2].shape, (3, 4))
        self.assertEqual(result[3].shape, (1, 4))  # Remainder

    def test_split_single_chunk(self):
        """Test splitting that results in single chunk

        When split_size equals the dimension size.
        """
        operation = SplitOperation(split_size=5, dim=0)

        x = torch.randn(5, 3)
        result = operation.forward(x=x)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (5, 3))
        self.assertTrue(torch.allclose(result[0], x))

    def test_split_size_validation(self):
        """Test that split_size validation works

        Should raise ValueError for invalid split sizes.
        """
        with self.assertRaises(ValueError) as context:
            SplitOperation(split_size=0)

        self.assertIn("split_size must be > 0", str(context.exception))

        with self.assertRaises(ValueError) as context:
            SplitOperation(split_size=-1)

        self.assertIn("split_size must be > 0", str(context.exception))

    def test_split_too_large_size(self):
        """Test splitting with size larger than dimension

        Should return the original tensor as single chunk.
        """
        operation = SplitOperation(split_size=10, dim=0)

        x = torch.randn(5, 3)  # Dimension size is 5
        result = operation.forward(x=x)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (5, 3))
        self.assertTrue(torch.allclose(result[0], x))

    def test_split_gradients(self):
        """Test that gradients flow correctly through split

        All output chunks should receive gradients.
        """
        operation = SplitOperation(split_size=2, dim=0)

        x = torch.randn(4, 3, requires_grad=True)
        result = operation.forward(x=x)

        # Create loss from all chunks
        loss = sum((chunk.sum() for chunk in result), torch.tensor(0.0))
        loss.backward()

        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_split_preserves_dtype(self):
        """Test that split preserves tensor dtype

        All output chunks should have the same dtype as input.
        """
        operation = SplitOperation(split_size=2, dim=1)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(3, 6, dtype=dtype)
                result = operation.forward(x=x)

                for chunk in result:
                    self.assertEqual(chunk.dtype, dtype)

    def test_split_negative_dimensions(self):
        """Test splitting along negative dimensions

        Should work with negative dimension indices.
        """
        operation = SplitOperation(split_size=3, dim=-1)

        x = torch.randn(4, 9)  # Last dimension has size 9
        result = operation.forward(x=x)

        # Should split along last dimension
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk.shape, (4, 3))

    def test_split_sequence_processing(self):
        """Test splitting for sequence processing

        Common pattern for processing long sequences in chunks.
        """
        operation = SplitOperation(split_size=4, dim=1)  # Split sequences

        batch_size, seq_len, hidden_dim = 2, 12, 32
        x = torch.randn(batch_size, seq_len, hidden_dim)

        result = operation.forward(x=x)

        # Should split into 3 chunks of length 4
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk.shape, (batch_size, 4, hidden_dim))

    def test_split_memory_efficiency(self):
        """Test that split creates views when possible

        Split should not copy data unnecessarily.
        """
        operation = SplitOperation(split_size=2, dim=0)

        x = torch.randn(4, 3)
        result = operation.forward(x=x)

        # Chunks should share memory with original tensor
        for i, chunk in enumerate(result):
            start_idx = i * 2
            end_idx = min((i + 1) * 2, x.shape[0])
            expected_chunk = x[start_idx:end_idx]
            self.assertEqual(chunk.data_ptr(), expected_chunk.data_ptr())

    def test_split_parameter_storage(self):
        """Test that split parameters are stored correctly

        The split_size and dim should be accessible as instance attributes.
        """
        operation = SplitOperation(split_size=3, dim=1)
        self.assertEqual(operation.split_size, 3)
        self.assertEqual(operation.dim, 1)

        operation_default = SplitOperation(split_size=2)  # dim defaults to -1
        self.assertEqual(operation_default.split_size, 2)
        self.assertEqual(operation_default.dim, -1)

    def test_split_empty_tensor(self):
        """Test splitting empty tensors

        Should handle edge cases gracefully.
        """
        operation = SplitOperation(split_size=2, dim=0)

        x = torch.empty(0, 3)  # Empty along split dimension
        result = operation.forward(x=x)

        # torch.split on empty tensor still returns one empty chunk
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (0, 3))

    def test_split_single_element(self):
        """Test splitting tensors with single elements

        Should work with minimal tensor sizes.
        """
        operation = SplitOperation(split_size=1, dim=0)

        x = torch.randn(3, 2)
        result = operation.forward(x=x)

        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk.shape, (1, 2))

    def test_split_large_tensors(self):
        """Test splitting large tensors

        Should handle larger tensors efficiently.
        """
        operation = SplitOperation(split_size=64, dim=0)

        x = torch.randn(256, 128)
        result = operation.forward(x=x)

        self.assertEqual(len(result), 4)  # 256 / 64 = 4
        for chunk in result:
            self.assertEqual(chunk.shape, (64, 128))

    def test_split_attention_patterns(self):
        """Test splitting patterns common in attention

        These are typical splits used in attention mechanisms.
        """
        # Split attention heads for separate processing
        operation = SplitOperation(split_size=1, dim=1)  # Split heads

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)

        result = operation.forward(x=x)

        self.assertEqual(len(result), num_heads)
        for chunk in result:
            self.assertEqual(chunk.shape, (batch_size, 1, seq_len, head_dim))

    def test_split_concat_round_trip(self):
        """Test that split followed by concat recovers original tensor

        This tests the inverse relationship between split and concat.
        """
        from caramba.operation.shape.concat import ConcatOperation

        split_op = SplitOperation(split_size=3, dim=0)
        concat_op = ConcatOperation(dim=0)

        x = torch.randn(9, 4)
        chunks = split_op.forward(x=x)
        result = concat_op.forward(*chunks)

        self.assertTrue(torch.allclose(result, x))

    def test_split_different_chunk_sizes(self):
        """Test that split handles uneven chunk sizes correctly

        Later chunks may be smaller than split_size.
        """
        operation = SplitOperation(split_size=7, dim=0)

        x = torch.randn(20, 3)  # 20 elements along dim 0
        result = operation.forward(x=x)

        # Should create chunks: [7, 7, 6]
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (7, 3))
        self.assertEqual(result[1].shape, (7, 3))
        self.assertEqual(result[2].shape, (6, 3))  # Remainder

        # Verify total elements
        total_elements = sum(chunk.shape[0] for chunk in result)
        self.assertEqual(total_elements, x.shape[0])