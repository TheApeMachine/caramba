"""Model weight initializers package."""

from caramba.initializers.base import Initializer
from caramba.initializers.gpt2 import GPT2Initializer

__all__ = [
    "Initializer",
    "GPT2Initializer",
]
