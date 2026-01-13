"""Test the base positional operation"""
from __future__ import annotations

import unittest

from caramba.operation.positional.base import PositionalOperation


class TestPositionalOperation(unittest.TestCase):
    """Test the base positional operation"""

    def test_inheritance(self):
        """Test that PositionalOperation properly inherits from Operation

        It should be an instance of Operation and PositionalOperation.
        """
        from caramba.operation.base import Operation

        operation = PositionalOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, PositionalOperation)

    def test_initialization(self):
        """Test that PositionalOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = PositionalOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)