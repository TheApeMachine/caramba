"""Test the base math operation"""
from __future__ import annotations

import unittest

from caramba.operation.math.base import MathOperation


class TestMathOperation(unittest.TestCase):
    """Test the base math operation"""

    def test_inheritance(self):
        """Test that MathOperation properly inherits from Operation

        It should be an instance of Operation and MathOperation.
        """
        from caramba.operation.base import Operation

        operation = MathOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, MathOperation)

    def test_initialization(self):
        """Test that MathOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = MathOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)