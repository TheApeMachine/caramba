"""Test the base cache operation"""
from __future__ import annotations

import unittest

from caramba.operation.cache.base import CacheOperation


class TestCacheOperation(unittest.TestCase):
    """Test the base cache operation"""

    def test_inheritance(self):
        """Test that CacheOperation properly inherits from Operation

        It should be an instance of Operation and CacheOperation.
        """
        from caramba.operation.base import Operation

        operation = CacheOperation()
        self.assertIsInstance(operation, Operation)
        self.assertIsInstance(operation, CacheOperation)

    def test_initialization(self):
        """Test that CacheOperation initializes correctly

        It should initialize without errors and be in training mode by default (like nn.Module).
        """
        operation = CacheOperation()
        self.assertIsNotNone(operation)
        # Should be in training mode by default (inherited from Operation -> nn.Module)
        self.assertTrue(operation.training)