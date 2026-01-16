"""Execution engines (backend selection).

Engines own backend-specific execution decisions (e.g. PyTorch device/dtype,
compilation, distributed wrappers) while keeping manifests backend-agnostic.
"""

from __future__ import annotations

from caramba.runtime.engine.torch_engine import TorchEngine

__all__ = ["TorchEngine", "get_mlx_engine"]


def get_mlx_engine():
    """Get MLX engine (lazy import to avoid errors on non-Apple platforms)."""
    from caramba.runtime.engine.mlx_engine import MLXEngine
    return MLXEngine()

