"""Test the base masking operation"""
from __future__ import annotations

import unittest

from caramba.operation.masking.base import MaskingOperation


class TestMaskingOperation(unittest.TestCase):
    """Test the base masking operation"""

    def test_inheritance(self):
        """Test that MaskingOperation properly inherits from Operation

        It should be an instance of Operation and MaskingOperation.
        """
        from caramba.operation.base import Operation

        operation = MaskingOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, MaskingOperation)

    def test_initialization(self):
        """Test that MaskingOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = MaskingOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)