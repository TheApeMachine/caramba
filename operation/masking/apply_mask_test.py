"""Test the Apply Mask operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.masking.apply_mask import ApplyMaskOperation


class TestApplyMaskOperation(unittest.TestCase):
    """Test the Apply Mask operation"""

    def test_basic_mask_application(self):
        """Test basic mask application to attention scores

        It should set masked positions to negative infinity.
        """
        operation = ApplyMaskOperation()

        # Create attention scores [batch, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len = 2, 4, 6
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Create mask that blocks the last two positions
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, -2:] = True  # Mask last 2 columns for each row

        result = operation.forward(scores=scores, mask=mask)

        # Check that masked positions are -inf
        masked_positions = mask
        self.assertTrue(torch.all(torch.isinf(result[masked_positions])))
        self.assertTrue(torch.all(result[masked_positions] < 0))  # Negative infinity

        # Check that unmasked positions remain unchanged
        unmasked_positions = ~mask
        self.assertTrue(torch.allclose(result[unmasked_positions], scores[unmasked_positions]))

    def test_mask_all_positions(self):
        """Test masking all positions

        All scores should become negative infinity.
        """
        operation = ApplyMaskOperation()

        scores = torch.randn(1, 2, 4, 4)
        mask = torch.ones_like(scores, dtype=torch.bool)  # Mask everything

        result = operation.forward(scores=scores, mask=mask)

        self.assertTrue(torch.all(torch.isinf(result)))
        self.assertTrue(torch.all(result < 0))

    def test_mask_no_positions(self):
        """Test masking no positions

        Scores should remain completely unchanged.
        """
        operation = ApplyMaskOperation()

        scores = torch.randn(1, 2, 4, 4)
        mask = torch.zeros_like(scores, dtype=torch.bool)  # Mask nothing

        result = operation.forward(scores=scores, mask=mask)

        self.assertTrue(torch.allclose(result, scores))

    def test_mask_different_shapes(self):
        """Test mask application with different tensor shapes

        It should handle various batch sizes, head counts, and sequence lengths.
        """
        operation = ApplyMaskOperation()

        test_configs = [
            (1, 1, 4, 4),   # Single batch, single head
            (3, 1, 4, 4),   # Multiple batches, single head
            (1, 8, 4, 4),   # Single batch, multiple heads
            (2, 4, 8, 8),   # Multiple batches and heads, longer sequence
        ]

        for batch_size, num_heads, seq_len, _ in test_configs:
            with self.subTest(batch_size=batch_size, num_heads=num_heads, seq_len=seq_len):
                scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
                mask = torch.randint(0, 2, (batch_size, num_heads, seq_len, seq_len), dtype=torch.bool)

                result = operation.forward(scores=scores, mask=mask)

                # Check shape
                self.assertEqual(result.shape, scores.shape)

                # Check that masked positions are -inf
                masked_positions = mask
                self.assertTrue(torch.all(torch.isinf(result[masked_positions])))

                # Check that unmasked positions are unchanged
                unmasked_positions = ~mask
                self.assertTrue(torch.allclose(result[unmasked_positions], scores[unmasked_positions]))

    def test_mask_preserves_gradients(self):
        """Test that mask application preserves gradient requirements

        If scores require gradients, the result should also require gradients.
        """
        operation = ApplyMaskOperation()

        scores = torch.randn(1, 2, 4, 4, requires_grad=True)
        mask = torch.randint(0, 2, (1, 2, 4, 4), dtype=torch.bool)

        result = operation.forward(scores=scores, mask=mask)

        self.assertTrue(result.requires_grad)

        # Test gradient flow
        loss = result.sum()
        loss.backward()

        assert scores.grad is not None

    def test_mask_different_dtypes(self):
        """Test mask application with different tensor dtypes

        It should work with various floating point precisions.
        """
        operation = ApplyMaskOperation()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                scores = torch.randn(1, 1, 4, 4, dtype=dtype)
                mask = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.bool)

                result = operation.forward(scores=scores, mask=mask)

                # Check dtype preservation
                self.assertEqual(result.dtype, dtype)

                # Check that masked positions are -inf
                masked_positions = mask
                self.assertTrue(torch.all(torch.isinf(result[masked_positions])))

    def test_mask_row_patterns(self):
        """Test masking entire rows (preventing attention to specific positions)

        This is common for padding masks where certain positions should never be attended to.
        """
        operation = ApplyMaskOperation()

        batch_size, num_heads, seq_len = 1, 1, 6
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Create mask that blocks attention to position 2 entirely
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, 2] = True  # Mask column 2 (attention to position 2)

        result = operation.forward(scores=scores, mask=mask)

        # All scores attending to position 2 should be -inf
        self.assertTrue(torch.all(torch.isinf(result[:, :, :, 2])))
        self.assertTrue(torch.all(result[:, :, :, 2] < 0))

        # Other positions should be unchanged
        other_positions = torch.zeros_like(mask)
        other_positions[:, :, :, 2] = True
        other_mask = ~other_positions
        self.assertTrue(torch.allclose(result[other_mask], scores[other_mask]))

    def test_mask_causal_pattern(self):
        """Test masking with causal pattern

        This simulates autoregressive attention where future positions are masked.
        """
        operation = ApplyMaskOperation()

        seq_len = 4
        scores = torch.randn(1, 1, seq_len, seq_len)

        # Create upper triangular mask (causal: can't attend to future)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        result = operation.forward(scores=scores, mask=mask)

        # Check that upper triangle (excluding diagonal) is -inf
        upper_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.all(torch.isinf(result[:, :, upper_mask])))
        self.assertTrue(torch.all(result[:, :, upper_mask] < 0))

        # Check that lower triangle (including diagonal) is unchanged
        lower_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.assertTrue(torch.allclose(result[:, :, lower_mask], scores[:, :, lower_mask]))

    def test_mask_symmetric_patterns(self):
        """Test masking with symmetric patterns

        This might be used for bidirectional attention with some positions masked.
        """
        operation = ApplyMaskOperation()

        seq_len = 4
        scores = torch.randn(1, 1, seq_len, seq_len)

        # Create symmetric mask (block attention between positions 1 and 3)
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, 1, 3] = True
        mask[:, :, 3, 1] = True

        result = operation.forward(scores=scores, mask=mask)

        # Check that the specified positions are masked
        self.assertTrue(torch.isinf(result[0, 0, 1, 3]))
        self.assertTrue(torch.isinf(result[0, 0, 3, 1]))

        # Check that other positions are unchanged
        other_mask = ~mask
        self.assertTrue(torch.allclose(result[other_mask], scores[other_mask]))

    def test_mask_broadcasting(self):
        """Test mask broadcasting with smaller mask dimensions

        It should handle cases where mask needs to be broadcasted.
        """
        operation = ApplyMaskOperation()

        # Full scores tensor
        scores = torch.randn(2, 4, 6, 6)

        # Smaller mask that should be broadcasted
        mask = torch.randint(0, 2, (6, 6), dtype=torch.bool)  # No batch/head dims

        result = operation.forward(scores=scores, mask=mask)

        # Should work through broadcasting
        self.assertEqual(result.shape, scores.shape)

        # Check that broadcasting worked correctly
        for b in range(2):
            for h in range(4):
                masked_positions = mask
                self.assertTrue(torch.all(torch.isinf(result[b, h, masked_positions])))

    def test_mask_numerical_stability(self):
        """Test numerical stability with extreme values

        Masking should work correctly even with very large or small score values.
        """
        operation = ApplyMaskOperation()

        # Test with very large scores
        scores_large = torch.randn(1, 1, 4, 4) * 1000
        mask = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.bool)

        result = operation.forward(scores=scores_large, mask=mask)

        # Masked positions should still be -inf
        masked_positions = mask
        self.assertTrue(torch.all(torch.isinf(result[masked_positions])))

        # Unmasked positions should be unchanged
        unmasked_positions = ~mask
        self.assertTrue(torch.allclose(result[unmasked_positions], scores_large[unmasked_positions]))

    def test_mask_empty_tensor(self):
        """Test masking with empty tensors

        It should handle edge cases with zero-sized dimensions.
        """
        operation = ApplyMaskOperation()

        # Test with zero sequence length
        scores = torch.empty(1, 1, 0, 0)
        mask = torch.empty(1, 1, 0, 0, dtype=torch.bool)

        result = operation.forward(scores=scores, mask=mask)

        self.assertEqual(result.shape, (1, 1, 0, 0))
        self.assertEqual(result.numel(), 0)

    def test_mask_boolean_consistency(self):
        """Test that mask interpretation is consistent

        True values should be masked (set to -inf), False values should be preserved.
        """
        operation = ApplyMaskOperation()

        scores = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1, 1, 2, 2]
        mask = torch.tensor([[[[True, False], [False, True]]]], dtype=torch.bool)

        result = operation.forward(scores=scores, mask=mask)

        # Position [0,0,0,0] should be -inf (masked)
        self.assertTrue(torch.isinf(result[0, 0, 0, 0]))
        self.assertTrue(result[0, 0, 0, 0] < 0)

        # Position [0,0,0,1] should be 2.0 (not masked)
        self.assertEqual(result[0, 0, 0, 1].item(), 2.0)

        # Position [0,0,1,0] should be 3.0 (not masked)
        self.assertEqual(result[0, 0, 1, 0].item(), 3.0)

        # Position [0,0,1,1] should be -inf (masked)
        self.assertTrue(torch.isinf(result[0, 0, 1, 1]))
        self.assertTrue(result[0, 0, 1, 1] < 0)