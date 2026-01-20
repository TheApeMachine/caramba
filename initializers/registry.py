"""Registry for weight initializers."""

from __future__ import annotations

from torch import nn

from config.weight_init import GPT2InitConfig, NoInitConfig, WeightInitConfig
from initializers.base import Initializer
from initializers.gpt2 import GPT2Initializer


class NoOpInitializer(Initializer):
    """Do nothing initializer."""

    def initialize(self, module: nn.Module) -> None:
        """Do nothing."""
        pass


class InitializerRegistry:
    """Registry to build initializers from config."""

    @staticmethod
    def build(config: WeightInitConfig) -> Initializer:
        """Build an initializer instance from configuration."""
        if isinstance(config, GPT2InitConfig):
            return GPT2Initializer(n_layers=config.n_layers)

        if isinstance(config, NoInitConfig):
            return NoOpInitializer()

        return NoOpInitializer()
