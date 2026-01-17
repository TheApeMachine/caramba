"""PyTorch adapters for caramba models.

This module provides PyTorch-specific utilities for:
- Attention surgery (replacing standard attention with DBA)
- Weight conversion and loading
"""

from .surgery import (
    AttentionSurgeryTorch,
    SurgeryConfig,
    run_surgery,
    stable_hash,
    xavier_uniform_,
    xavier_uniform_tensor,
)

__all__ = [
    "AttentionSurgeryTorch",
    "SurgeryConfig",
    "run_surgery",
    "stable_hash",
    "xavier_uniform_",
    "xavier_uniform_tensor",
]
