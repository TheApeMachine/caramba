"""Model weight initializers package."""

from initializers.base import Initializer
from initializers.gpt2 import GPT2Initializer

__all__ = [
    "Initializer",
    "GPT2Initializer",
]
