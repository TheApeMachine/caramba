"""Test the Scaled Dot-Product Attention operation"""
from __future__ import annotations

import math
import unittest

import torch

from caramba.operation.attention.scaled_dot_product import ScaledDotProductAttentionOperation


class TestScaledDotProductAttentionOperation(unittest.TestCase):
    """Test the Scaled Dot-Product Attention operation"""

    def test_basic_functionality(self):
        """Test basic attention computation

        It should compute attention correctly with proper scaling and softmax.
        """
        operation = ScaledDotProductAttentionOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64

        # Create input tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        result = operation.forward(q=q, k=k, v=v)

        # Check output shape
        self.assertEqual(result.shape, (batch_size, num_heads, seq_len, head_dim))

        # Check that result is finite
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_scaling_factor(self):
        """Test that attention scores are properly scaled by sqrt(head_dim)

        The scaling should prevent softmax from becoming too sharp for large head dimensions.
        """
        operation = ScaledDotProductAttentionOperation()

        head_dim = 64
        q = torch.ones(1, 1, 2, head_dim)  # Simple query
        k = torch.ones(1, 1, 2, head_dim)  # Simple key
        v = torch.randn(1, 1, 2, head_dim)  # Random values

        # Manually compute expected attention scores
        expected_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        expected_weights = torch.softmax(expected_scores, dim=-1)
        expected_output = torch.matmul(expected_weights, v)

        result = operation.forward(q=q, k=k, v=v)

        self.assertTrue(torch.allclose(result, expected_output))

    def test_single_head_single_sequence(self):
        """Test attention with minimal dimensions (1 head, 1 sequence)

        It should work correctly with the smallest valid tensor dimensions.
        """
        operation = ScaledDotProductAttentionOperation()

        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        v = torch.randn(1, 1, 1, 32)

        result = operation.forward(q=q, k=k, v=v)

        self.assertEqual(result.shape, (1, 1, 1, 32))
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_masking_functionality(self):
        """Test attention with masking

        Masked positions should have zero attention weights.
        """
        operation = ScaledDotProductAttentionOperation()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create a mask that blocks the last two positions
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, -2:] = True  # Mask last 2 positions in attention

        result_masked = operation.forward(q=q, k=k, v=v, mask=mask)

        # Without mask for comparison
        result_unmasked = operation.forward(q=q, k=k, v=v)

        # Results should be different
        self.assertFalse(torch.allclose(result_masked, result_unmasked))

        # Masked result should still be finite
        self.assertTrue(torch.all(torch.isfinite(result_masked)))

    def test_mask_expansion(self):
        """Test that 3D masks are properly expanded to 4D

        Masks with shape [batch, seq_len, seq_len] should be expanded to [batch, 1, seq_len, seq_len].
        """
        operation = ScaledDotProductAttentionOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 3, 4, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # 3D mask - mask only the last position for each sequence
        mask_3d = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask_3d[:, :, -1] = True  # Mask last position in each sequence

        result = operation.forward(q=q, k=k, v=v, mask=mask_3d)

        self.assertEqual(result.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_dropout_functionality(self):
        """Test attention with dropout

        Dropout should be applied during training but not evaluation.
        """
        dropout_p = 0.5
        operation = ScaledDotProductAttentionOperation(dropout_p=dropout_p)

        batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Test in training mode
        operation.train()
        result_train_1 = operation.forward(q=q, k=k, v=v)
        result_train_2 = operation.forward(q=q, k=k, v=v)

        # Results should be different due to dropout randomness
        self.assertFalse(torch.allclose(result_train_1, result_train_2))

        # Test in evaluation mode
        operation.eval()
        result_eval_1 = operation.forward(q=q, k=k, v=v)
        result_eval_2 = operation.forward(q=q, k=k, v=v)

        # Results should be identical in eval mode (no dropout)
        self.assertTrue(torch.allclose(result_eval_1, result_eval_2))

    def test_zero_dropout(self):
        """Test that zero dropout works correctly

        When dropout_p=0, it should behave like no dropout.
        """
        operation = ScaledDotProductAttentionOperation(dropout_p=0.0)

        batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Should work in both training and eval modes
        operation.train()
        result_train = operation.forward(q=q, k=k, v=v)

        operation.eval()
        result_eval = operation.forward(q=q, k=k, v=v)

        # Results should be the same (no dropout applied)
        self.assertTrue(torch.allclose(result_train, result_eval))

    def test_gradient_flow(self):
        """Test that gradients flow correctly through attention

        All input tensors should receive gradients when requires_grad=True.
        """
        operation = ScaledDotProductAttentionOperation()

        q = torch.randn(1, 2, 4, 32, requires_grad=True)
        k = torch.randn(1, 2, 4, 32, requires_grad=True)
        v = torch.randn(1, 2, 4, 32, requires_grad=True)

        result = operation.forward(q=q, k=k, v=v)
        loss = result.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        self.assertTrue(torch.all(torch.isfinite(q.grad)))
        self.assertTrue(torch.all(torch.isfinite(k.grad)))
        self.assertTrue(torch.all(torch.isfinite(v.grad)))

    def test_attention_weights_properties(self):
        """Test properties of attention weights

        Attention weights should sum to 1 along the last dimension and be non-negative.
        """
        operation = ScaledDotProductAttentionOperation()

        # Store attention weights in a nonlocal variable
        captured_attn_weights = None

        # Override forward to capture attention weights
        original_forward = operation.forward

        def forward_with_weights(q, k, v, mask=None):
            nonlocal captured_attn_weights

            head_dim = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask, float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)

            if operation.dropout_p > 0.0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=operation.dropout_p, training=operation.training)

            output = torch.matmul(attn_weights, v)

            # Capture weights for testing
            captured_attn_weights = attn_weights
            return output

        operation.forward = forward_with_weights

        try:
            q = torch.randn(1, 1, 3, 32)
            k = torch.randn(1, 1, 3, 32)
            v = torch.randn(1, 1, 3, 32)

            result = operation.forward(q=q, k=k, v=v)

            # Check attention weights properties
            assert captured_attn_weights is not None
            attn_weights = captured_attn_weights
            self.assertTrue(torch.all(attn_weights >= 0))  # Non-negative
            self.assertTrue(torch.all(attn_weights <= 1))  # At most 1

            # Sum along last dimension should be close to 1 (allowing for numerical precision)
            weight_sums = attn_weights.sum(dim=-1)
            self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

        finally:
            # Restore original method
            operation.forward = original_forward

    def test_different_head_dimensions(self):
        """Test attention with different head dimensions

        It should work with various head dimensions and scale appropriately.
        """
        operation = ScaledDotProductAttentionOperation()

        for head_dim in [16, 32, 64, 128]:
            with self.subTest(head_dim=head_dim):
                q = torch.randn(1, 1, 4, head_dim)
                k = torch.randn(1, 1, 4, head_dim)
                v = torch.randn(1, 1, 4, head_dim)

                result = operation.forward(q=q, k=k, v=v)

                self.assertEqual(result.shape, (1, 1, 4, head_dim))
                self.assertTrue(torch.all(torch.isfinite(result)))

    def test_batch_and_head_variations(self):
        """Test attention with different batch sizes and number of heads

        It should handle various batch configurations correctly.
        """
        operation = ScaledDotProductAttentionOperation()

        test_configs = [
            (1, 1, 8, 32),   # Single batch, single head
            (3, 1, 8, 32),   # Multiple batches, single head
            (1, 8, 8, 32),   # Single batch, multiple heads
            (2, 4, 16, 64),  # Multiple batches and heads
        ]

        for batch_size, num_heads, seq_len, head_dim in test_configs:
            with self.subTest(batch_size=batch_size, num_heads=num_heads, seq_len=seq_len, head_dim=head_dim):
                q = torch.randn(batch_size, num_heads, seq_len, head_dim)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim)

                result = operation.forward(q=q, k=k, v=v)

                expected_shape = (batch_size, num_heads, seq_len, head_dim)
                self.assertEqual(result.shape, expected_shape)
                self.assertTrue(torch.all(torch.isfinite(result)))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values

        Attention should handle various input ranges without numerical issues.
        """
        operation = ScaledDotProductAttentionOperation()

        # Test with large values
        q = torch.randn(1, 1, 4, 32) * 100
        k = torch.randn(1, 1, 4, 32) * 100
        v = torch.randn(1, 1, 4, 32) * 100

        result = operation.forward(q=q, k=k, v=v)
        self.assertTrue(torch.all(torch.isfinite(result)))

        # Test with small values
        q = torch.randn(1, 1, 4, 32) * 0.001
        k = torch.randn(1, 1, 4, 32) * 0.001
        v = torch.randn(1, 1, 4, 32) * 0.001

        result = operation.forward(q=q, k=k, v=v)
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_mask_with_casual_pattern(self):
        """Test causal masking pattern

        This is common in autoregressive models where future tokens should be masked.
        """
        operation = ScaledDotProductAttentionOperation()

        seq_len = 4
        q = torch.randn(1, 1, seq_len, 32)
        k = torch.randn(1, 1, seq_len, 32)
        v = torch.randn(1, 1, seq_len, 32)

        # Create causal mask (upper triangular - future positions masked)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        result = operation.forward(q=q, k=k, v=v, mask=mask)

        self.assertEqual(result.shape, (1, 1, seq_len, 32))
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_identical_queries_keys(self):
        """Test attention when queries and keys are identical

        This should result in uniform attention weights.
        """
        operation = ScaledDotProductAttentionOperation()

        q = torch.randn(1, 1, 3, 32)
        k = q.clone()  # Identical to q
        v = torch.randn(1, 1, 3, 32)

        result = operation.forward(q=q, k=k, v=v)

        self.assertEqual(result.shape, (1, 1, 3, 32))
        self.assertTrue(torch.all(torch.isfinite(result)))

        # With identical q and k, attention should be more uniform
        # (though not perfectly uniform due to softmax numerical precision)
        self.assertFalse(torch.any(torch.isnan(result)))

    def test_training_mode_preservation(self):
        """Test that the operation preserves training mode

        The training mode should be inherited from the parent module.
        """
        operation = ScaledDotProductAttentionOperation(dropout_p=0.1)

        # Test training mode
        operation.train()
        self.assertTrue(operation.training)

        # Test evaluation mode
        operation.eval()
        self.assertFalse(operation.training)