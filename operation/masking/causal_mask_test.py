"""Test the Causal Mask operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.masking.causal_mask import CausalMaskOperation


class TestCausalMaskOperation(unittest.TestCase):
    """Test the Causal Mask operation"""

    def test_basic_causal_mask(self):
        """Test basic causal mask creation

        It should create an upper triangular mask where future positions are masked.
        """
        seq_len = 4
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))

        # Check that it's boolean
        self.assertEqual(mask.dtype, torch.bool)

        # Check the triangular structure
        # Position (i,j) should be True if j > i (future positions)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertTrue(mask[i, j], f"Position ({i},{j}) should be masked")
                else:
                    self.assertFalse(mask[i, j], f"Position ({i},{j}) should not be masked")

    def test_causal_mask_batch_expansion(self):
        """Test causal mask with batch expansion

        It should expand the mask for multiple batches.
        """
        seq_len = 3
        batch_size = 2
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward(batch_size=batch_size)

        # Check shape
        expected_shape = (batch_size, seq_len, seq_len)
        self.assertEqual(mask.shape, expected_shape)

        # Check that all batches have the same mask pattern
        for batch in range(batch_size):
            batch_mask = mask[batch]
            for i in range(seq_len):
                for j in range(seq_len):
                    expected = j > i
                    self.assertEqual(batch_mask[i, j].item(), expected,
                                   f"Batch {batch}, position ({i},{j}) incorrect")

    def test_causal_mask_different_sequence_lengths(self):
        """Test causal masks with different sequence lengths

        It should create correctly sized masks for various sequence lengths.
        """
        for seq_len in [1, 2, 5, 10, 16]:
            with self.subTest(seq_len=seq_len):
                operation = CausalMaskOperation(seq_len=seq_len)

                mask = operation.forward()

                # Check shape
                self.assertEqual(mask.shape, (seq_len, seq_len))

                # Check triangular structure
                for i in range(seq_len):
                    for j in range(seq_len):
                        expected = j > i
                        self.assertEqual(mask[i, j].item(), expected,
                                       f"Seq len {seq_len}, position ({i},{j}) incorrect")

    def test_causal_mask_single_position(self):
        """Test causal mask for sequence length 1

        It should create a 1x1 mask with no positions masked.
        """
        operation = CausalMaskOperation(seq_len=1)

        mask = operation.forward()

        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(mask[0, 0])  # No positions should be masked

    def test_causal_mask_two_positions(self):
        """Test causal mask for sequence length 2

        It should mask only the future position.
        """
        operation = CausalMaskOperation(seq_len=2)

        mask = operation.forward()

        self.assertEqual(mask.shape, (2, 2))

        # Position (0,0): can attend to itself -> False
        self.assertFalse(mask[0, 0])

        # Position (0,1): can attend to future -> True (masked)
        self.assertTrue(mask[0, 1])

        # Position (1,0): can attend to past -> False
        self.assertFalse(mask[1, 0])

        # Position (1,1): can attend to itself -> False
        self.assertFalse(mask[1, 1])

    def test_causal_mask_properties(self):
        """Test mathematical properties of causal masks

        The mask should be upper triangular with diagonal zeros.
        """
        seq_len = 5
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # Check that it's upper triangular (above diagonal is True)
        upper_triangular = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.allclose(mask, upper_triangular))

        # Check that diagonal is all False (can attend to current position)
        diagonal = torch.diag(mask)
        self.assertTrue(torch.all(~diagonal))

        # Check that lower triangle is all False (can attend to past)
        lower_triangle = torch.tril(mask, diagonal=-1)
        self.assertTrue(torch.all(~lower_triangle))

    def test_causal_mask_autoregressive_property(self):
        """Test that causal mask enforces autoregressive property

        Each position can only attend to itself and previous positions.
        """
        seq_len = 6
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # For each position i, it can attend to positions 0 through i
        for i in range(seq_len):
            # Positions 0 to i should not be masked
            can_attend = ~mask[i, :i+1]
            self.assertTrue(torch.all(can_attend),
                          f"Position {i} should be able to attend to positions 0-{i}")

            # Positions i+1 to end should be masked
            if i + 1 < seq_len:
                masked = mask[i, i+1:]
                self.assertTrue(torch.all(masked),
                              f"Position {i} should not attend to positions {i+1}+")

    def test_causal_mask_batch_consistency(self):
        """Test that all batches get identical causal masks

        When expanded for batches, all batch elements should have the same mask.
        """
        seq_len = 4
        batch_size = 3
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward(batch_size=batch_size)

        # All batches should be identical
        for batch in range(1, batch_size):
            self.assertTrue(torch.allclose(mask[0], mask[batch]),
                          f"Batch {batch} differs from batch 0")

    def test_causal_mask_memory_efficiency(self):
        """Test that causal mask creation is memory efficient

        It should create the expected triangular pattern without unnecessary computation.
        """
        seq_len = 100  # Large sequence for testing
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # Verify it's still correct despite large size
        self.assertEqual(mask.shape, (seq_len, seq_len))

        # Check a few key positions
        self.assertFalse(mask[0, 0])  # Can attend to self
        self.assertTrue(mask[0, 1])   # Cannot attend to future
        self.assertFalse(mask[5, 3])  # Can attend to past
        self.assertTrue(mask[3, 5])   # Cannot attend to future

    def test_causal_mask_no_expansion_needed(self):
        """Test causal mask when batch_size=1

        It should return a 2D tensor without unnecessary expansion.
        """
        seq_len = 3
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward(batch_size=1)

        # Should be 2D when batch_size=1
        self.assertEqual(mask.shape, (seq_len, seq_len))

        # The pattern should be correct
        expected = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.allclose(mask, expected))

    def test_causal_mask_large_batch_sizes(self):
        """Test causal mask with large batch sizes

        It should handle batch expansion efficiently.
        """
        seq_len = 4
        batch_size = 8
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward(batch_size=batch_size)

        self.assertEqual(mask.shape, (batch_size, seq_len, seq_len))

        # All batches should have the same pattern
        expected_2d = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        for batch in range(batch_size):
            self.assertTrue(torch.allclose(mask[batch], expected_2d))

    def test_causal_mask_compatibility_with_attention(self):
        """Test that causal masks are compatible with attention operations

        The mask should work correctly when used with attention scores.
        """
        from caramba.operation.masking.apply_mask import ApplyMaskOperation

        seq_len = 4
        causal_op = CausalMaskOperation(seq_len=seq_len)
        apply_op = ApplyMaskOperation()

        # Create attention scores
        batch_size, num_heads = 1, 1
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Create causal mask and expand for attention
        causal_mask = causal_op.forward(batch_size=batch_size)
        if causal_mask.dim() == 2:  # If 2D, expand to 3D first
            causal_mask = causal_mask.unsqueeze(0)
        causal_mask = causal_mask.unsqueeze(1)  # Add head dimension

        # Apply the mask
        masked_scores = apply_op.forward(scores=scores, mask=causal_mask)

        # Verify that future positions are masked (-inf)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertTrue(torch.isinf(masked_scores[0, 0, i, j]))
                self.assertTrue(masked_scores[0, 0, i, j] < 0)

        # Verify that past and current positions are not additionally masked
        # (they may have been -inf in original scores, but shouldn't be -inf due to masking)
        for i in range(seq_len):
            for j in range(i + 1):
                # If the original score was not -inf, it should remain unchanged
                if not torch.isinf(scores[0, 0, i, j]):
                    self.assertEqual(masked_scores[0, 0, i, j], scores[0, 0, i, j])

    def test_causal_mask_sequence_ordering(self):
        """Test that mask respects sequence position ordering

        Position i should be able to attend to positions 0 through i, but not i+1 through end.
        """
        seq_len = 5
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # Each row represents what one position can attend to
        for position in range(seq_len):
            row = mask[position]

            # Should not attend to future positions (right side of diagonal)
            future_positions = row[position + 1:]
            if len(future_positions) > 0:
                self.assertTrue(torch.all(future_positions),
                              f"Position {position} should mask all future positions")

            # Should attend to current and past positions (left side including diagonal)
            past_positions = row[:position + 1]
            self.assertTrue(torch.all(~past_positions),
                          f"Position {position} should not mask current and past positions")

    def test_causal_mask_strict_upper_triangular(self):
        """Test that causal mask is strictly upper triangular

        Only positions above the diagonal should be masked.
        """
        seq_len = 6
        operation = CausalMaskOperation(seq_len=seq_len)

        mask = operation.forward()

        # Check that it's exactly the upper triangular matrix with diagonal=1
        expected = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.allclose(mask, expected))

        # Verify no extra positions are masked
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    self.assertFalse(mask[i, j], f"Position ({i},{j}) should not be masked")
                else:
                    self.assertTrue(mask[i, j], f"Position ({i},{j}) should be masked")