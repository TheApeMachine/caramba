"""Base Adapter class

Defines the common interface for all adapters.
"""
from  __future__ import annotations

from config.topology import (
    NestedTopologyConfig,
    StackedTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
)



class BaseAdapter:
    def __init__(self, model: str):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        return "Hello, world!"
        
