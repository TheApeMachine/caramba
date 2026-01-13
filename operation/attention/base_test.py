"""Test the base attention operation"""
from __future__ import annotations

import unittest

from caramba.operation.attention.base import AttentionOperation


class TestAttentionOperation(unittest.TestCase):
    """Test the base attention operation"""

    def test_inheritance(self):
        """Test that AttentionOperation properly inherits from Operation

        It should be an instance of Operation and AttentionOperation.
        """
        from caramba.operation.base import Operation

        operation = AttentionOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, AttentionOperation)

    def test_initialization(self):
        """Test that AttentionOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = AttentionOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)