"""Test the base activation operation"""
from __future__ import annotations

import unittest

import torch

from caramba.operation.activation.base import ActivationOperation


class TestActivationOperation(unittest.TestCase):
    """Test the base activation operation"""
    def test_forward(self):
        """Test the forward pass

        It should raise a NotImplementedError.
        """
        operation = ActivationOperation()
        with self.assertRaises(NotImplementedError):
            operation.forward(x=torch.tensor([1, 2, 3]))