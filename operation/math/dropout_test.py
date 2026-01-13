"""Test the Dropout operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.math.dropout import DropoutOperation


class TestDropoutOperation(unittest.TestCase):
    """Test the Dropout operation"""

    def test_dropout_training_mode(self):
        """Test dropout in training mode

        It should randomly zero out elements with the specified probability.
        """
        torch.manual_seed(42)  # For reproducible results
        operation = DropoutOperation(p=0.5, training=True)

        x = torch.ones(1000, 10)  # Large tensor for statistical testing
        result = operation.forward(x=x)

        # Check that some elements are zeroed out
        zeros_count = (result == 0).sum().item()
        self.assertGreater(zeros_count, 0)

        # Check that remaining elements are scaled (dropout scales by 1/(1-p))
        nonzero_values = result[result != 0]
        expected_scale = 1.0 / (1.0 - 0.5)  # = 2.0
        self.assertTrue(torch.allclose(nonzero_values, torch.full_like(nonzero_values, expected_scale)))

    def test_dropout_eval_mode(self):
        """Test dropout in evaluation mode

        It should return the input unchanged.
        """
        operation = DropoutOperation(p=0.5, training=False)

        x = torch.randn(5, 10)
        result = operation.forward(x=x)

        self.assertTrue(torch.allclose(result, x))

    def test_dropout_zero_probability(self):
        """Test dropout with zero probability

        It should return input unchanged in both training and eval modes.
        """
        x = torch.randn(3, 4)

        for training in [True, False]:
            operation = DropoutOperation(p=0.0, training=training)
            result = operation.forward(x=x)
            self.assertTrue(torch.allclose(result, x))

    def test_dropout_full_probability(self):
        """Test dropout with full probability (p=1.0)

        In training mode, all elements should be zeroed. In eval mode, unchanged.
        """
        x = torch.ones(5, 5)

        # Training mode
        operation_train = DropoutOperation(p=1.0, training=True)
        result_train = operation_train.forward(x=x)
        self.assertTrue(torch.all(result_train == 0))

        # Eval mode
        operation_eval = DropoutOperation(p=1.0, training=False)
        result_eval = operation_eval.forward(x=x)
        self.assertTrue(torch.allclose(result_eval, x))

    def test_dropout_probability_bounds(self):
        """Test dropout probability bounds

        It should accept probabilities between 0 and 1.
        """
        # Valid probabilities
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            operation = DropoutOperation(p=p)
            self.assertEqual(operation.p, p)

        # Invalid probabilities
        for p in [-0.1, 1.1, 2.0]:
            with self.assertRaises(ValueError):
                DropoutOperation(p=p)

    def test_dropout_gradients_training(self):
        """Test gradients through dropout in training mode

        Gradients should flow through non-dropped elements.
        """
        torch.manual_seed(42)
        operation = DropoutOperation(p=0.0, training=True)  # No dropout for gradient test

        x = torch.randn(4, 4, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_dropout_gradients_eval(self):
        """Test gradients through dropout in evaluation mode

        Gradients should flow normally.
        """
        operation = DropoutOperation(p=0.5, training=False)

        x = torch.randn(3, 3, requires_grad=True)
        result = operation.forward(x=x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_dropout_different_dtypes(self):
        """Test dropout with different tensor dtypes

        It should work with various floating point precisions.
        """
        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                operation = DropoutOperation(p=0.2, training=False)

                x = torch.ones(5, 5, dtype=dtype)
                result = operation.forward(x=x)

                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.allclose(result, x))

    def test_dropout_preserves_shape(self):
        """Test that dropout preserves tensor shape

        Output should have the same shape as input.
        """
        operation = DropoutOperation(p=0.3, training=True)

        shapes = [(10,), (5, 8), (2, 3, 4), (2, 3, 4, 5)]

        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                result = operation.forward(x=x)
                self.assertEqual(result.shape, x.shape)

    def test_dropout_deterministic_eval(self):
        """Test that dropout is deterministic in evaluation mode

        Multiple calls should give identical results.
        """
        operation = DropoutOperation(p=0.5, training=False)

        x = torch.randn(10, 10)

        result1 = operation.forward(x=x)
        result2 = operation.forward(x=x)

        self.assertTrue(torch.allclose(result1, result2))

    def test_dropout_random_training(self):
        """Test that dropout is random in training mode

        Multiple calls should typically give different results.
        """
        torch.manual_seed(42)
        operation = DropoutOperation(p=0.5, training=True)

        x = torch.ones(100, 100)  # Large tensor to ensure randomness

        result1 = operation.forward(x=x)
        result2 = operation.forward(x=x)

        # Results should be different (with very high probability)
        self.assertFalse(torch.allclose(result1, result2))

    def test_dropout_scaling_factor(self):
        """Test that dropout applies correct scaling

        In training mode, surviving elements should be scaled by 1/(1-p).
        """
        torch.manual_seed(42)
        p = 0.5
        operation = DropoutOperation(p=p, training=True)

        x = torch.ones(1000, 1)  # Large tensor for statistical testing
        result = operation.forward(x=x)

        # Find non-zero elements
        nonzero_mask = result != 0
        if nonzero_mask.any():
            nonzero_values = result[nonzero_mask]
            expected_scale = 1.0 / (1.0 - p)  # = 2.0
            self.assertTrue(torch.allclose(nonzero_values, torch.full_like(nonzero_values, expected_scale)))

    def test_dropout_parameter_access(self):
        """Test that dropout parameters are accessible

        The p and training parameters should be stored as instance attributes.
        """
        operation = DropoutOperation(p=0.3, training=True)
        self.assertEqual(operation.p, 0.3)
        self.assertEqual(operation.training, True)

        operation2 = DropoutOperation(p=0.7, training=False)
        self.assertEqual(operation2.p, 0.7)
        self.assertEqual(operation2.training, False)