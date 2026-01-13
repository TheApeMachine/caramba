"""Test the base shape operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.shape.base import ShapeOperation


class TestShapeOperation(unittest.TestCase):
    """Test the base shape operation"""

    def test_inheritance(self):
        """Test that ShapeOperation properly inherits from Operation

        It should be an instance of Operation and ShapeOperation.
        """
        from caramba.operation.base import Operation

        operation = ShapeOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, ShapeOperation)

    def test_initialization(self):
        """Test that ShapeOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = ShapeOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError

        The base class should not implement forward.
        """
        operation = ShapeOperation()

        x = torch.randn(2, 3)
        with self.assertRaises(NotImplementedError):
            operation.forward(x=x)