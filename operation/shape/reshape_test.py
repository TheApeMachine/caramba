"""Test the Reshape operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.reshape import ReshapeOperation


class TestReshapeOperation(unittest.TestCase):
    """Test the Reshape operation"""

    def test_basic_reshape(self):
        """Test basic reshape functionality

        It should change tensor shape while preserving total elements.
        """
        operation = ReshapeOperation(shape=(6, 4))

        x = torch.randn(2, 12)  # 24 elements
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (6, 4))
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_with_inference(self):
        """Test reshape with -1 for dimension inference

        Should infer the correct dimension size.
        """
        operation = ReshapeOperation(shape=(4, -1))

        x = torch.randn(2, 12)  # 24 elements
        result = operation.forward(x=x)

        # Should infer second dimension as 24/4 = 6
        self.assertEqual(result.shape, (4, 6))
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_multidimensional(self):
        """Test reshape with complex multidimensional changes

        Should handle various shape transformations.
        """
        x = torch.randn(2, 3, 4)  # 24 elements

        test_shapes = [
            (24,),      # Flatten
            (4, 6),     # 2D
            (2, 12),    # Different 2D
            (2, 2, 6),  # 3D
            (1, 24),    # With singleton dimension
            (24, 1),    # Different singleton
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                operation = ReshapeOperation(shape=shape)
                result = operation.forward(x=x)

                self.assertEqual(result.shape, shape)
                self.assertEqual(result.numel(), x.numel())

    def test_reshape_attention_patterns(self):
        """Test reshape patterns common in attention mechanisms

        These are typical reshapes used in transformers.
        """
        # Simulate attention workflow reshapes
        batch_size, seq_len, hidden_dim = 2, 8, 64

        # Start with [batch, seq_len, hidden_dim]
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Flatten for some computation
        operation = ReshapeOperation(shape=(batch_size * seq_len, hidden_dim))
        result = operation.forward(x=x)

        expected_shape = (batch_size * seq_len, hidden_dim)
        self.assertEqual(result.shape, expected_shape)

        # Reshape back
        operation_back = ReshapeOperation(shape=(batch_size, seq_len, hidden_dim))
        result_back = operation_back.forward(x=result)

        self.assertEqual(result_back.shape, x.shape)
        self.assertTrue(torch.allclose(result_back, x))

    def test_reshape_gradients(self):
        """Test that gradients flow correctly through reshape

        Reshape should preserve gradient flow.
        """
        operation = ReshapeOperation(shape=(6, 4))

        x = torch.randn(2, 12, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertEqual(x.grad.shape, x.shape)

    def test_reshape_preserves_dtype(self):
        """Test that reshape preserves tensor dtype

        Output should have the same dtype as input.
        """
        operation = ReshapeOperation(shape=(8, 3))

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 12, dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)

    def test_reshape_memory_efficiency(self):
        """Test that reshape creates views when possible

        Reshape should not copy data unnecessarily.
        """
        operation = ReshapeOperation(shape=(12, 2))

        x = torch.randn(3, 8)
        result = operation.forward(x=x)

        # For compatible reshapes, should be a view
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_invalid_total_elements(self):
        """Test reshape with incompatible total element count

        Should raise an error when shapes are incompatible.
        """
        operation = ReshapeOperation(shape=(7, 5))  # 35 elements

        x = torch.randn(2, 12)  # 24 elements

        with self.assertRaises(RuntimeError):
            operation.forward(x=x)

    def test_reshape_parameter_storage(self):
        """Test that reshape shape parameter is stored correctly

        The shape parameter should be accessible as an instance attribute.
        """
        shape = (4, 6, 2)
        operation = ReshapeOperation(shape=shape)
        self.assertEqual(operation.shape, shape)

    def test_reshape_single_element(self):
        """Test reshape with single element tensors

        Should work with minimal tensors.
        """
        operation = ReshapeOperation(shape=(1,))

        x = torch.randn(1, 1, 1)
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.numel(), 1)

    def test_reshape_large_tensors(self):
        """Test reshape with large tensors

        Should handle larger tensors efficiently.
        """
        operation = ReshapeOperation(shape=(1024, 256))

        x = torch.randn(512, 512)
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (1024, 256))
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_complex_inference(self):
        """Test reshape with multiple -1 dimensions

        Should infer the correct sizes for multiple unknown dimensions.
        """
        # This should work in PyTorch (only one -1 allowed)
        operation = ReshapeOperation(shape=(4, -1))

        x = torch.randn(2, 12)  # 24 elements
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (4, 6))  # 24 / 4 = 6

    def test_reshape_identity(self):
        """Test reshape to the same shape

        Should return an equivalent tensor.
        """
        x = torch.randn(3, 4, 5)
        operation = ReshapeOperation(shape=x.shape)

        result = operation.forward(x=x)

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.allclose(result, x))

    def test_reshape_flatten(self):
        """Test reshaping to flattened 1D tensor

        Common pattern for linear layers.
        """
        operation = ReshapeOperation(shape=(-1,))

        x = torch.randn(2, 3, 4, 5)
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (120,))  # 2*3*4*5 = 120
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_add_dimensions(self):
        """Test reshape that adds dimensions

        Should add singleton dimensions as needed.
        """
        operation = ReshapeOperation(shape=(2, 3, 4, 1))

        x = torch.randn(2, 12)  # 24 elements
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (2, 3, 4, 1))
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_remove_dimensions(self):
        """Test reshape that removes dimensions

        Should collapse dimensions appropriately.
        """
        operation = ReshapeOperation(shape=(24,))

        x = torch.randn(2, 3, 4)  # 24 elements
        result = operation.forward(x=x)

        self.assertEqual(result.shape, (24,))
        self.assertEqual(result.numel(), x.numel())

    def test_reshape_batch_processing(self):
        """Test reshape in batch processing scenarios

        Should work correctly with batched data.
        """
        batch_size, seq_len, features = 4, 10, 32

        # Flatten features for linear layer
        operation = ReshapeOperation(shape=(batch_size * seq_len, features))

        x = torch.randn(batch_size, seq_len, features)
        result = operation.forward(x=x)

        expected_shape = (batch_size * seq_len, features)
        self.assertEqual(result.shape, expected_shape)

        # Reshape back
        operation_back = ReshapeOperation(shape=(batch_size, seq_len, features))
        result_back = operation_back.forward(x=result)

        self.assertEqual(result_back.shape, x.shape)
        self.assertTrue(torch.allclose(result_back, x))