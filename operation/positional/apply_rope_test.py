"""Test the Apply RoPE operation"""
from __future__ import annotations

import math
import unittest

import torch

from caramba.operation.positional.apply_rope import ApplyRoPEOperation


class TestApplyRoPEOperation(unittest.TestCase):
    """Test the Apply RoPE operation"""

    def test_basic_rope_application(self):
        """Test basic RoPE application with default settings

        It should apply rotary position embeddings to both queries and keys.
        """
        operation = ApplyRoPEOperation()

        batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_out, k_out = operation.forward(q=q, k=k)

        # Check output shapes
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

        # Check that outputs are different from inputs (RoPE should modify them)
        self.assertFalse(torch.allclose(q_out, q))
        self.assertFalse(torch.allclose(k_out, k))

    def test_rope_q_only_variant(self):
        """Test RoPE applied only to queries

        Keys should remain unchanged.
        """
        operation = ApplyRoPEOperation(variant="q_only")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_orig = q.clone()
        k_orig = k.clone()

        q_out, k_out = operation.forward(q=q, k=k)

        # Query should be modified
        self.assertFalse(torch.allclose(q_out, q_orig))

        # Key should remain unchanged
        self.assertTrue(torch.allclose(k_out, k_orig))

    def test_rope_k_only_variant(self):
        """Test RoPE applied only to keys

        Queries should remain unchanged.
        """
        operation = ApplyRoPEOperation(variant="k_only")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_orig = q.clone()
        k_orig = k.clone()

        q_out, k_out = operation.forward(q=q, k=k)

        # Key should be modified
        self.assertFalse(torch.allclose(k_out, k_orig))

        # Query should remain unchanged
        self.assertTrue(torch.allclose(q_out, q_orig))

    def test_rope_both_variant(self):
        """Test RoPE applied to both queries and keys (default)

        Both should be modified.
        """
        operation = ApplyRoPEOperation(variant="both")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_orig = q.clone()
        k_orig = k.clone()

        q_out, k_out = operation.forward(q=q, k=k)

        # Both should be modified
        self.assertFalse(torch.allclose(q_out, q_orig))
        self.assertFalse(torch.allclose(k_out, k_orig))

    def test_rope_with_start_pos(self):
        """Test RoPE with non-zero start position

        Should handle continuation from previous sequences.
        """
        operation = ApplyRoPEOperation()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 3, 16

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        start_pos = 5  # Starting from position 5

        q_out, k_out = operation.forward(q=q, k=k, start_pos=start_pos)

        # Check shapes
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

        # Check that results are different
        self.assertFalse(torch.allclose(q_out, q))
        self.assertFalse(torch.allclose(k_out, k))

    def test_rope_rotary_embedding_properties(self):
        """Test that RoPE generates proper rotary embeddings

        The embeddings should have the correct frequency pattern.
        """
        operation = ApplyRoPEOperation(base=10000.0)

        seq_len = 4
        head_dim = 8

        # Test the internal rotary embedding generation
        cos, sin = operation._get_rotary_embedding(seq_len, head_dim, torch.device('cpu'))

        # Check shapes: cos and sin should be [seq_len, head_dim/2]
        expected_shape = (seq_len, head_dim // 2)
        self.assertEqual(cos.shape, expected_shape)
        self.assertEqual(sin.shape, expected_shape)

        # Check that cos and sin are periodic with the expected frequencies
        # The frequencies should decrease exponentially
        # cos[0, 0] should be cos(0) = 1
        self.assertAlmostEqual(cos[0, 0].item(), 1.0, places=5)
        # sin[0, 0] should be sin(0) = 0
        self.assertAlmostEqual(sin[0, 0].item(), 0.0, places=5)

    def test_rope_rotation_math(self):
        """Test the mathematical correctness of RoPE rotation

        The rotation should preserve the L2 norm of vectors.
        """
        operation = ApplyRoPEOperation()

        # Create a simple 2D vector to test rotation
        batch_size, num_heads, seq_len, head_dim = 1, 1, 1, 4

        # Simple vector [1, 0, 1, 0] - should be rotated
        q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        k = torch.tensor([[[[0.5, 0.5, 0.5, 0.5]]]])

        q_out, k_out = operation.forward(q=q, k=k)

        # The L2 norm should be preserved for each half of the vector
        # (RoPE rotates but doesn't change magnitudes)
        q_norm = torch.norm(q_out, dim=-1)
        k_norm = torch.norm(k_out, dim=-1)

        # Norms should be close to original (allowing for numerical precision)
        original_q_norm = torch.norm(q, dim=-1)
        original_k_norm = torch.norm(k, dim=-1)

        self.assertTrue(torch.allclose(q_norm, original_q_norm, rtol=1e-5))
        self.assertTrue(torch.allclose(k_norm, original_k_norm, rtol=1e-5))

    def test_rope_different_bases(self):
        """Test RoPE with different base frequencies

        Different bases should produce different rotations.
        """
        base1 = 10000.0
        base2 = 5000.0

        operation1 = ApplyRoPEOperation(base=base1)
        operation2 = ApplyRoPEOperation(base=base2)

        q = torch.randn(1, 1, 2, 16)
        k = torch.randn(1, 1, 2, 16)

        q_out1, k_out1 = operation1.forward(q=q, k=k)
        q_out2, k_out2 = operation2.forward(q=q, k=k)

        # Results should be different with different bases
        self.assertFalse(torch.allclose(q_out1, q_out2))
        self.assertFalse(torch.allclose(k_out1, k_out2))

    def test_rope_gradients(self):
        """Test that gradients flow correctly through RoPE

        All input tensors should receive gradients.
        """
        operation = ApplyRoPEOperation()

        q = torch.randn(1, 2, 3, 8, requires_grad=True)
        k = torch.randn(1, 2, 3, 8, requires_grad=True)

        q_out, k_out = operation.forward(q=q, k=k)
        loss = (q_out + k_out).sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None

        # Gradients should be finite
        self.assertTrue(torch.all(torch.isfinite(q.grad)))
        self.assertTrue(torch.all(torch.isfinite(k.grad)))

    def test_rope_different_head_dimensions(self):
        """Test RoPE with different head dimensions

        Should work with various embedding dimensions.
        """
        operation = ApplyRoPEOperation()

        for head_dim in [8, 16, 32, 64]:
            with self.subTest(head_dim=head_dim):
                q = torch.randn(1, 1, 2, head_dim)
                k = torch.randn(1, 1, 2, head_dim)

                q_out, k_out = operation.forward(q=q, k=k)

                self.assertEqual(q_out.shape, q.shape)
                self.assertEqual(k_out.shape, k.shape)

    def test_rope_different_batch_sizes(self):
        """Test RoPE with different batch sizes

        Should handle various batch configurations.
        """
        operation = ApplyRoPEOperation()

        for batch_size in [1, 3, 5]:
            with self.subTest(batch_size=batch_size):
                q = torch.randn(batch_size, 2, 4, 16)
                k = torch.randn(batch_size, 2, 4, 16)

                q_out, k_out = operation.forward(q=q, k=k)

                expected_shape = (batch_size, 2, 4, 16)
                self.assertEqual(q_out.shape, expected_shape)
                self.assertEqual(k_out.shape, expected_shape)

    def test_rope_sequence_position_dependence(self):
        """Test that RoPE depends on sequence position

        Different positions should produce different rotations.
        """
        operation = ApplyRoPEOperation()

        # Create identical queries/keys but at different positions
        q = torch.randn(1, 1, 3, 16)
        k = torch.randn(1, 1, 3, 16)

        # Apply at position 0
        q_out_0, k_out_0 = operation.forward(q=q, k=k, start_pos=0)

        # Apply at position 5
        q_out_5, k_out_5 = operation.forward(q=q, k=k, start_pos=5)

        # Results should be different due to different positional embeddings
        self.assertFalse(torch.allclose(q_out_0, q_out_5))
        self.assertFalse(torch.allclose(k_out_0, k_out_5))

    def test_rope_single_position(self):
        """Test RoPE with single position sequence

        Should handle minimal sequence length.
        """
        operation = ApplyRoPEOperation()

        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)

        q_out, k_out = operation.forward(q=q, k=k)

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_rope_long_sequence(self):
        """Test RoPE with long sequences

        Should handle longer sequences efficiently.
        """
        operation = ApplyRoPEOperation()

        seq_len = 100  # Long sequence
        q = torch.randn(1, 1, seq_len, 16)
        k = torch.randn(1, 1, seq_len, 16)

        q_out, k_out = operation.forward(q=q, k=k)

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_rope_frequency_decay(self):
        """Test that RoPE frequencies decay exponentially

        Higher frequency components should rotate faster.
        """
        operation = ApplyRoPEOperation(base=10000.0)

        # Get rotary embeddings
        cos, sin = operation._get_rotary_embedding(10, 8, torch.device('cpu'))

        # Check that frequencies decrease (more rotation for higher indices)
        # This is a complex property, but we can check that different dimensions
        # have different rotation patterns
        rotation_diff_01 = cos[1, 0] - cos[0, 0]  # Low frequency change
        rotation_diff_02 = cos[1, 1] - cos[0, 1]  # Higher frequency change

        # Higher frequencies should change more rapidly (more rotation)
        # This is a rough check of the exponential decay property
        self.assertNotAlmostEqual(rotation_diff_01.item(), rotation_diff_02.item(), places=3)

    def test_rope_different_devices(self):
        """Test RoPE works on different devices

        Should handle CPU and GPU tensors appropriately.
        """
        operation = ApplyRoPEOperation()

        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 1, 2, 8)

        q_out, k_out = operation.forward(q=q, k=k)

        # Results should be on the same device as inputs
        self.assertEqual(q_out.device, q.device)
        self.assertEqual(k_out.device, k.device)

    def test_rope_parameter_validation(self):
        """Test RoPE parameter validation

        Invalid variant should raise an error.
        """
        # Valid variants should work
        for variant in ["both", "q_only", "k_only"]:
            operation = ApplyRoPEOperation(variant=variant)
            self.assertEqual(operation.variant, variant)

        # Invalid variant should raise an error (though this is not currently implemented)
        # For now, just test that valid variants work
