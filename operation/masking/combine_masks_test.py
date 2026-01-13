"""Test the Combine Masks operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.masking.combine_masks import CombineMasksOperation


class TestCombineMasksOperation(unittest.TestCase):
    """Test the Combine Masks operation"""

    def test_combine_two_masks(self):
        """Test combining two masks with logical OR

        The result should be True where either input mask is True.
        """
        operation = CombineMasksOperation()

        # Create two masks [batch, seq_len, seq_len]
        batch_size, seq_len = 1, 4
        mask1 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask1[0, 0, 1] = True  # Mask position (0,1)

        mask2 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask2[0, 2, 3] = True  # Mask position (2,3)

        result = operation.forward(masks=[mask1, mask2])

        # Check shape
        self.assertEqual(result.shape, (batch_size, seq_len, seq_len))

        # Check that both positions are masked in result
        self.assertTrue(result[0, 0, 1])  # From mask1
        self.assertTrue(result[0, 2, 3])  # From mask2

        # Check that other positions are not masked
        self.assertFalse(result[0, 0, 0])
        self.assertFalse(result[0, 1, 2])
        self.assertFalse(result[0, 3, 3])

    def test_combine_three_masks(self):
        """Test combining three masks

        The result should be True where any of the three masks is True.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 3
        mask1 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask1[0, 0, 1] = True

        mask2 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask2[0, 1, 2] = True

        mask3 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask3[0, 2, 0] = True

        result = operation.forward(masks=[mask1, mask2, mask3])

        # All three positions should be masked
        self.assertTrue(result[0, 0, 1])
        self.assertTrue(result[0, 1, 2])
        self.assertTrue(result[0, 2, 0])

        # Other positions should not be masked
        self.assertFalse(result[0, 0, 0])
        self.assertFalse(result[0, 1, 1])

    def test_combine_overlapping_masks(self):
        """Test combining masks with overlapping True positions

        Overlapping positions should still result in True.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 3
        mask1 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask1[0, 1, 1] = True
        mask1[0, 1, 2] = True

        mask2 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask2[0, 1, 2] = True  # Overlaps with mask1
        mask2[0, 2, 2] = True

        result = operation.forward(masks=[mask1, mask2])

        # Position (1,1) should be True (from mask1)
        self.assertTrue(result[0, 1, 1])

        # Position (1,2) should be True (from both masks)
        self.assertTrue(result[0, 1, 2])

        # Position (2,2) should be True (from mask2)
        self.assertTrue(result[0, 2, 2])

    def test_combine_empty_masks(self):
        """Test combining masks where some masks are all False

        The result should be correct even with empty masks.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 3
        mask1 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)  # All False
        mask2 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask2[0, 1, 1] = True  # One True

        result = operation.forward(masks=[mask1, mask2])

        # Should be identical to mask2
        self.assertTrue(torch.allclose(result, mask2))

    def test_combine_single_mask(self):
        """Test combining a single mask

        It should return the input mask unchanged.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 4
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)

        result = operation.forward(masks=[mask])

        # Should be identical to input
        self.assertTrue(torch.allclose(result, mask))

    def test_combine_many_masks(self):
        """Test combining many masks (stress test)

        It should handle combining multiple masks correctly.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 4
        num_masks = 5

        masks = []
        for i in range(num_masks):
            mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
            mask[0, i % seq_len, i % seq_len] = True  # Each mask sets one diagonal element
            masks.append(mask)

        result = operation.forward(masks=masks)

        # All diagonal positions should be True
        for i in range(min(num_masks, seq_len)):
            self.assertTrue(result[0, i, i],
                          f"Diagonal position ({i},{i}) should be True")

    def test_combine_broadcasting(self):
        """Test combining masks with broadcasting

        Masks with compatible shapes should broadcast correctly.
        """
        operation = CombineMasksOperation()

        # Test broadcasting: (1, 4, 4) and (4, 4) -> (1, 4, 4)
        mask1 = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask1[0, 0, 1] = True

        mask2 = torch.zeros(4, 4, dtype=torch.bool)
        mask2[2, 3] = True

        result = operation.forward(masks=[mask1, mask2])

        # Should broadcast to (1, 4, 4)
        self.assertEqual(result.shape, (1, 4, 4))

        # Should have both positions set
        self.assertTrue(result[0, 0, 1])
        self.assertTrue(result[0, 2, 3])

    def test_combine_no_masks_error(self):
        """Test that combining no masks raises an error

        At least one mask must be provided.
        """
        operation = CombineMasksOperation()

        with self.assertRaises(ValueError) as context:
            operation.forward(masks=[])

        self.assertIn("At least one mask must be provided", str(context.exception))

    def test_combine_preserves_dtype(self):
        """Test that combining masks preserves boolean dtype

        The result should always be boolean.
        """
        operation = CombineMasksOperation()

        mask1 = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask2 = torch.ones(1, 3, 3, dtype=torch.bool)

        result = operation.forward(masks=[mask1, mask2])

        self.assertEqual(result.dtype, torch.bool)

    def test_combine_logical_or_property(self):
        """Test that combination follows logical OR semantics

        Result[i,j] should be True if any input mask[i,j] is True.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 1, 3

        # Create masks with specific patterns
        mask1 = torch.tensor([[[True, False, True],
                              [False, True, False],
                              [True, False, False]]], dtype=torch.bool)

        mask2 = torch.tensor([[[False, True, True],
                              [True, False, False],
                              [False, True, True]]], dtype=torch.bool)

        result = operation.forward(masks=[mask1, mask2])

        expected = torch.tensor([[[True, True, True],   # True | False, False | True, True | True
                                 [True, True, False],   # False | True, True | False, False | False
                                 [True, True, True]]], dtype=torch.bool)  # True | False, False | True, False | True

        self.assertTrue(torch.allclose(result, expected))

    def test_combine_commutative(self):
        """Test that mask combination is commutative

        Order of masks should not affect the result.
        """
        operation = CombineMasksOperation()

        mask1 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        mask2 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        mask3 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)

        result_123 = operation.forward(masks=[mask1, mask2, mask3])
        result_132 = operation.forward(masks=[mask1, mask3, mask2])
        result_213 = operation.forward(masks=[mask2, mask1, mask3])
        result_231 = operation.forward(masks=[mask2, mask3, mask1])
        result_312 = operation.forward(masks=[mask3, mask1, mask2])
        result_321 = operation.forward(masks=[mask3, mask2, mask1])

        # All combinations should be identical
        self.assertTrue(torch.allclose(result_123, result_132))
        self.assertTrue(torch.allclose(result_123, result_213))
        self.assertTrue(torch.allclose(result_123, result_231))
        self.assertTrue(torch.allclose(result_123, result_312))
        self.assertTrue(torch.allclose(result_123, result_321))

    def test_combine_associative(self):
        """Test that mask combination is associative

        Grouping should not affect the result: (a|b)|c = a|(b|c)
        """
        operation = CombineMasksOperation()

        mask1 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        mask2 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        mask3 = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)

        # (mask1 | mask2) | mask3
        temp = operation.forward(masks=[mask1, mask2])
        result_left = operation.forward(masks=[temp, mask3])

        # mask1 | (mask2 | mask3)
        temp = operation.forward(masks=[mask2, mask3])
        result_right = operation.forward(masks=[mask1, temp])

        self.assertTrue(torch.allclose(result_left, result_right))

    def test_combine_idempotent(self):
        """Test that combining identical masks is idempotent

        Combining a mask with itself should give the same mask.
        """
        operation = CombineMasksOperation()

        mask = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)

        result = operation.forward(masks=[mask, mask])

        self.assertTrue(torch.allclose(result, mask))

    def test_combine_identity_element(self):
        """Test that combining with all-False mask is identity operation

        Combining any mask with an all-False mask should give the original mask.
        """
        operation = CombineMasksOperation()

        mask = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        identity = torch.zeros_like(mask)  # All False

        result = operation.forward(masks=[mask, identity])

        self.assertTrue(torch.allclose(result, mask))

    def test_combine_dominance_element(self):
        """Test that combining with all-True mask gives all-True

        Combining any mask with an all-True mask should give all-True.
        """
        operation = CombineMasksOperation()

        mask = torch.randint(0, 2, (1, 4, 4), dtype=torch.bool)
        dominance = torch.ones_like(mask)  # All True

        result = operation.forward(masks=[mask, dominance])

        self.assertTrue(torch.all(result))

    def test_combine_batch_consistency(self):
        """Test that combination works consistently across batch dimensions

        Each batch element should be processed independently.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 3, 4

        mask1 = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)
        mask2 = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)

        result = operation.forward(masks=[mask1, mask2])

        # Check each batch element individually
        for batch in range(batch_size):
            batch_result = result[batch]
            batch_mask1 = mask1[batch]
            batch_mask2 = mask2[batch]

            expected = batch_mask1 | batch_mask2
            self.assertTrue(torch.allclose(batch_result, expected))

    def test_combine_large_tensors(self):
        """Test combining masks with large tensor dimensions

        It should handle larger tensors efficiently.
        """
        operation = CombineMasksOperation()

        batch_size, seq_len = 2, 64  # Larger dimensions

        mask1 = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)
        mask2 = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)
        mask3 = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)

        result = operation.forward(masks=[mask1, mask2, mask3])

        expected = mask1 | mask2 | mask3
        self.assertTrue(torch.allclose(result, expected))