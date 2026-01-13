"""Test the Concat operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.concat import ConcatOperation


class TestConcatOperation(unittest.TestCase):
    """Test the Concat operation"""

    def test_basic_concatenation(self):
        """Test basic concatenation along the last dimension

        It should join tensors along dimension -1 by default.
        """
        operation = ConcatOperation()

        a = torch.tensor([[1, 2], [3, 4]])  # [2, 2]
        b = torch.tensor([[5, 6], [7, 8]])  # [2, 2]

        result = operation.forward(a, b)

        expected = torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]])  # [2, 4]
        self.assertTrue(torch.allclose(result, expected))

    def test_concat_along_different_dimensions(self):
        """Test concatenation along different dimensions

        Should work along any valid dimension.
        """
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)

        for dim in [0, 1, 2]:
            with self.subTest(dim=dim):
                operation = ConcatOperation(dim=dim)
                result = operation.forward(a, b)

                expected = torch.cat([a, b], dim=dim)
                self.assertTrue(torch.allclose(result, expected))

    def test_concat_multiple_tensors(self):
        """Test concatenation of multiple tensors

        Should handle more than two tensors.
        """
        operation = ConcatOperation()

        a = torch.tensor([[1], [2]])  # [2, 1]
        b = torch.tensor([[3], [4]])  # [2, 1]
        c = torch.tensor([[5], [6]])  # [2, 1]

        result = operation.forward(a, b, c)

        expected = torch.tensor([[1, 3, 5], [2, 4, 6]])  # [2, 3]
        self.assertTrue(torch.allclose(result, expected))

    def test_concat_single_tensor(self):
        """Test concatenation with single tensor

        Should return the tensor unchanged.
        """
        operation = ConcatOperation()

        a = torch.randn(3, 4)
        result = operation.forward(a)

        self.assertTrue(torch.allclose(result, a))

    def test_concat_empty_inputs(self):
        """Test concatenation with no inputs

        Should raise a ValueError from torch.cat.
        """
        operation = ConcatOperation()

        with self.assertRaises(ValueError) as context:
            operation.forward()

        self.assertIn("expected a non-empty list of Tensors", str(context.exception))

    def test_concat_different_shapes_compatible(self):
        """Test concatenation with compatible different shapes

        Should work when all dimensions except the concat dimension match.
        """
        operation = ConcatOperation(dim=0)

        a = torch.randn(2, 3, 4)
        b = torch.randn(1, 3, 4)  # Different size in dim 0
        c = torch.randn(3, 3, 4)  # Different size in dim 0

        result = operation.forward(a, b, c)

        expected = torch.cat([a, b, c], dim=0)
        self.assertEqual(result.shape, (6, 3, 4))  # 2+1+3=6
        self.assertTrue(torch.allclose(result, expected))

    def test_concat_different_shapes_incompatible(self):
        """Test concatenation with incompatible shapes

        Should raise an error when non-concat dimensions don't match.
        """
        operation = ConcatOperation(dim=0)

        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 4)  # Different size in dim 1

        with self.assertRaises(RuntimeError):
            operation.forward(a, b)

    def test_concat_gradients(self):
        """Test that gradients flow correctly through concatenation

        All input tensors should receive gradients in their respective regions.
        """
        operation = ConcatOperation(dim=1)

        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 4, requires_grad=True)

        result = operation.forward(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

        # Check that gradients have correct shapes
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

    def test_concat_preserves_dtype(self):
        """Test that concatenation preserves tensor dtype

        Output should have the same dtype as inputs.
        """
        operation = ConcatOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                a = torch.randn(2, 3, dtype=dtype)
                b = torch.randn(2, 4, dtype=dtype)

                result = operation.forward(a, b)

                self.assertEqual(result.dtype, dtype)

    def test_concat_negative_dimensions(self):
        """Test concatenation along negative dimensions

        Should work with negative dimension indices.
        """
        a = torch.randn(3, 4, 5)
        b = torch.randn(3, 4, 6)

        # Concat along last dimension (-1)
        operation = ConcatOperation(dim=-1)
        result = operation.forward(a, b)

        expected = torch.cat([a, b], dim=-1)
        self.assertTrue(torch.allclose(result, expected))

        # Concat along second-to-last dimension (-2)
        operation = ConcatOperation(dim=-2)
        c = torch.randn(3, 2, 5)
        result = operation.forward(a, c)

        expected = torch.cat([a, c], dim=-2)
        self.assertTrue(torch.allclose(result, expected))

    def test_concat_attention_patterns(self):
        """Test concatenation patterns common in attention mechanisms

        These are typical concat operations used in transformers.
        """
        # Test sequence concatenation (for longer contexts)
        operation = ConcatOperation(dim=1)  # Sequence dimension

        batch_size, seq_len, hidden_dim = 2, 4, 64

        seq1 = torch.randn(batch_size, seq_len, hidden_dim)
        seq2 = torch.randn(batch_size, seq_len, hidden_dim)

        result = operation.forward(seq1, seq2)

        expected_shape = (batch_size, seq_len * 2, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

        # Test feature concatenation (for multi-head or feature fusion)
        operation = ConcatOperation(dim=-1)  # Feature dimension

        features1 = torch.randn(batch_size, seq_len, hidden_dim // 2)
        features2 = torch.randn(batch_size, seq_len, hidden_dim // 2)

        result = operation.forward(features1, features2)

        expected_shape = (batch_size, seq_len, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_concat_memory_efficiency(self):
        """Test that concatenation creates contiguous tensors

        Concat should produce contiguous results.
        """
        operation = ConcatOperation()

        a = torch.randn(2, 3)
        b = torch.randn(2, 4)

        result = operation.forward(a, b)

        # Result should be contiguous
        self.assertTrue(result.is_contiguous())

    def test_concat_parameter_validation(self):
        """Test that concat dimension parameter is stored correctly

        The dim parameter should be accessible as an instance attribute.
        """
        operation = ConcatOperation(dim=2)
        self.assertEqual(operation.dim, 2)

        operation_default = ConcatOperation()
        self.assertEqual(operation_default.dim, -1)  # Default is last dimension

    def test_concat_edge_cases(self):
        """Test concatenation with edge cases

        Should handle unusual but valid concatenation scenarios.
        """
        # Test with zero-sized dimensions
        operation = ConcatOperation(dim=1)

        a = torch.randn(2, 0, 3)  # Zero size in concat dimension
        b = torch.randn(2, 4, 3)

        result = operation.forward(a, b)
        self.assertEqual(result.shape, (2, 4, 3))

        # Test single element tensors
        operation = ConcatOperation(dim=0)

        a = torch.randn(1, 2, 3)
        b = torch.randn(1, 2, 3)

        result = operation.forward(a, b)
        self.assertEqual(result.shape, (2, 2, 3))

    def test_concat_large_tensors(self):
        """Test concatenation with large tensors

        Should handle larger computations efficiently.
        """
        operation = ConcatOperation(dim=0)

        # Test with reasonably large tensors
        a = torch.randn(128, 64)
        b = torch.randn(256, 64)

        result = operation.forward(a, b)

        expected_shape = (384, 64)  # 128 + 256 = 384
        self.assertEqual(result.shape, expected_shape)

    def test_concat_different_devices_error(self):
        """Test concatenation with tensors on different devices

        Should raise an error when tensors are on different devices.
        """
        if torch.cuda.is_available():
            operation = ConcatOperation()

            a = torch.randn(2, 3).cuda()
            b = torch.randn(2, 4).cpu()

            with self.assertRaises(RuntimeError):
                operation.forward(a, b)

    def test_concat_mixed_dtypes_error(self):
        """Test concatenation with different dtypes

        PyTorch allows this, so we just verify it works.
        """
        operation = ConcatOperation()

        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(2, 4, dtype=torch.float64)

        # PyTorch will upcast to the wider type
        result = operation.forward(a, b)
        self.assertEqual(result.dtype, torch.float64)