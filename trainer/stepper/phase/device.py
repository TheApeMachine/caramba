"""Device synchronization helpers for training loops."""

from __future__ import annotations

import torch


class DeviceSynchronizer:
    """Device synchronizer

    Training loops often measure wall-clock timings (throughput, step latency).
    On asynchronous devices (CUDA/MPS), timings are only meaningful when we
    synchronize at well-defined boundaries.
    """

    def sync(self, *, device: torch.device) -> None:
        """Synchronize the configured device, when supported."""
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
            return
        if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()  # type: ignore[attr-defined]

