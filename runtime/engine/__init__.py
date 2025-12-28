"""Execution engines (backend selection).

Engines own backend-specific execution decisions (e.g. PyTorch device/dtype,
compilation, distributed wrappers) while keeping manifests backend-agnostic.
"""

from __future__ import annotations

from runtime.engine.torch_engine import TorchEngine

__all__ = ["TorchEngine"]

