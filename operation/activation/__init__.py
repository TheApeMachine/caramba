"""Activation function operations

Non-linear activation functions that introduce non-linearity into neural networks,
enabling them to learn complex patterns and representations.
"""
from __future__ import annotations

from caramba.operation.activation.base import ActivationOperation
from caramba.operation.activation.gelu import GELUOperation
from caramba.operation.activation.leaky_relu import LeakyReLUOperation
from caramba.operation.activation.relu import ReLUOperation
from caramba.operation.activation.sigmoid import SigmoidOperation
from caramba.operation.activation.swiglu import SwiGLUOperation
from caramba.operation.activation.tanh import TanhOperation

__all__ = [
    "ActivationOperation",
    "GELUOperation",
    "LeakyReLUOperation",
    "ReLUOperation",
    "SigmoidOperation",
    "SwiGLUOperation",
    "TanhOperation",
]