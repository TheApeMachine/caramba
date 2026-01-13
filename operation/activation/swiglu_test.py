"""Test the SwiGLU activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.swiglu import SwiGLUOperation


class TestSwiGLUOperation(unittest.TestCase):
    """Test the SwiGLU activation operation"""

    def test_forward_basic_functionality(self):
        """Test the basic forward pass functionality

        It should split input in half and apply gating: x1 * silu(x2).
        """
        operation = SwiGLUOperation()

        # Input tensor with even last dimension (6 elements, split into 3 and 3)
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = operation.forward(x=input_tensor)

        # Expected: x1 * silu(x2) where x1 = [1, 2, 3], x2 = [4, 5, 6]
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        expected = x1 * torch.nn.functional.silu(x2)

        self.assertEqual(result.shape, (1, 3))  # Output should be half the size
        self.assertTrue(torch.allclose(result.squeeze(), expected))

    def test_forward_gating_logic(self):
        """Test that the gating logic works correctly

        SwiGLU should implement: output = x1 * silu(x2) where input is split [x1, x2].
        """
        operation = SwiGLUOperation()

        # Create a simple test case where we know the expected values
        x1 = torch.tensor([2.0, 3.0])
        x2 = torch.tensor([0.0, 1.0])  # silu(0) = 0, silu(1) ≈ 0.731
        input_tensor = torch.cat([x1, x2], dim=-1).unsqueeze(0)  # Shape: [1, 4]

        result = operation.forward(x=input_tensor)

        # Expected: [2.0 * silu(0), 3.0 * silu(1)] = [2.0 * 0, 3.0 * ~0.731]
        silu_x2 = torch.nn.functional.silu(x2)
        expected = x1 * silu_x2

        self.assertTrue(torch.allclose(result.squeeze(), expected))

    def test_forward_dimension_reduction(self):
        """Test that output dimension is half of input dimension

        SwiGLU reduces the last dimension by half.
        """
        operation = SwiGLUOperation()

        # Test various input shapes with even last dimensions
        test_cases = [
            (torch.randn(2, 6), (2, 3)),      # [2, 6] -> [2, 3]
            (torch.randn(3, 4, 8), (3, 4, 4)), # [3, 4, 8] -> [3, 4, 4]
            (torch.randn(1, 10), (1, 5)),     # [1, 10] -> [1, 5]
        ]

        for input_tensor, expected_shape in test_cases:
            with self.subTest(input_shape=input_tensor.shape):
                result = operation.forward(x=input_tensor)
                self.assertEqual(result.shape, expected_shape)

    def test_forward_multidimensional_tensor(self):
        """Test SwiGLU with complex multidimensional tensors

        It should only split along the last dimension.
        """
        operation = SwiGLUOperation()

        # Create a 3D tensor: [batch, seq, hidden*2]
        input_tensor = torch.randn(2, 3, 8)  # Last dim 8 will become 4
        result = operation.forward(x=input_tensor)

        self.assertEqual(result.shape, (2, 3, 4))

        # Verify the gating logic element-wise
        for i in range(2):
            for j in range(3):
                x1 = input_tensor[i, j, :4]  # First half
                x2 = input_tensor[i, j, 4:]  # Second half
                expected = x1 * torch.nn.functional.silu(x2)
                self.assertTrue(torch.allclose(result[i, j], expected))

    def test_forward_different_dtypes(self):
        """Test the forward pass with different tensor dtypes

        It should work with float16, float32, and float64.
        """
        operation = SwiGLUOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
                result = operation.forward(x=input_tensor)

                # Check that output has same dtype
                self.assertEqual(result.dtype, dtype)

                # Check that shape is correct (4 -> 2)
                self.assertEqual(result.shape, (1, 2))

    def test_forward_requires_grad(self):
        """Test that the operation preserves gradient requirements

        The output should require gradients if input requires gradients.
        """
        operation = SwiGLUOperation()

        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        result = operation.forward(x=input_tensor)

        self.assertTrue(result.requires_grad)

        # Test backward pass
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)

    def test_forward_gradient_flow(self):
        """Test gradient flow through SwiGLU

        Gradients should flow correctly through both x1 and x2 components.
        """
        operation = SwiGLUOperation()

        # Create input with gradient tracking
        x = torch.tensor([[1.0, 0.0, 0.5, 1.0]], requires_grad=True)  # [x1=1.0, x1=0.0, x2=0.5, x2=1.0]
        y = operation.forward(x=x)
        y.backward(torch.ones_like(y))

        # Gradient should exist
        self.assertIsNotNone(x.grad)

        # The gradient should be non-zero (complex interaction between x1 and silu(x2))
        self.assertFalse(torch.all(x.grad == 0))

    def test_forward_with_various_values(self):
        """Test SwiGLU with various input value ranges

        It should handle positive, negative, and zero values correctly.
        """
        operation = SwiGLUOperation()

        # Test with mixed values
        input_tensor = torch.tensor([[1.0, -1.0, 0.0, 2.0]])  # x1 = [1, -1], x2 = [0, 2]
        result = operation.forward(x=input_tensor)

        x1 = torch.tensor([1.0, -1.0])
        x2 = torch.tensor([0.0, 2.0])
        expected = x1 * torch.nn.functional.silu(x2)

        self.assertTrue(torch.allclose(result.squeeze(), expected))

    def test_forward_silu_properties(self):
        """Test that SwiGLU uses SiLU (Swish) for gating

        SiLU(x) = x * sigmoid(x), and should be smooth and differentiable.
        """
        operation = SwiGLUOperation()

        # Test that when x2 is large positive, gate is approximately x2
        # When x2 is large negative, gate approaches 0
        large_pos = torch.tensor([[1.0, 1.0, 10.0, 10.0]])
        result_pos = operation.forward(x=large_pos)

        large_neg = torch.tensor([[1.0, 1.0, -10.0, -10.0]])
        result_neg = operation.forward(x=large_neg)

        # For large positive x2, result should be approximately x1 * x2
        self.assertTrue(torch.allclose(result_pos, torch.tensor([[10.0, 10.0]]), rtol=0.1))

        # For large negative x2, result should be close to 0
        self.assertTrue(torch.all(result_neg.abs() < 1e-3))

    def test_forward_batch_consistency(self):
        """Test that SwiGLU works consistently across batch dimensions

        Each batch element should be processed independently.
        """
        operation = SwiGLUOperation()

        # Create batch of different inputs
        batch_input = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],  # batch 1
            [2.0, 3.0, 4.0, 5.0],  # batch 2
        ])

        result = operation.forward(x=batch_input)

        # Process each batch element separately
        result_0 = operation.forward(x=batch_input[0:1])
        result_1 = operation.forward(x=batch_input[1:2])

        self.assertTrue(torch.allclose(result[0:1], result_0))
        self.assertTrue(torch.allclose(result[1:2], result_1))

    def test_forward_output_properties(self):
        """Test mathematical properties of SwiGLU output

        The output should have the correct shape and be finite.
        """
        operation = SwiGLUOperation()

        # Test with normal values
        input_tensor = torch.randn(3, 6)
        result = operation.forward(x=input_tensor)

        # Check shape
        self.assertEqual(result.shape, (3, 3))

        # Check finiteness
        self.assertTrue(torch.all(torch.isfinite(result)))

        # Check that result can be zero (when either x1=0 or silu(x2)=0)
        zero_input = torch.tensor([[0.0, 1.0, 1.0, 0.0]])  # x1=0, x2=1 or x1=1, x2=0
        zero_result = operation.forward(x=zero_input)
        self.assertTrue(torch.allclose(zero_result, torch.zeros_like(zero_result)))