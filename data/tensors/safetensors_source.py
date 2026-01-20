"""Safetensors tensor source

Lazily loads tensors from safetensors files, caching the loaded tensor
and length to avoid repeated file opens. Thread-safe for DataLoader usage.
"""
from __future__ import annotations

import threading
from pathlib import Path

from torch import Tensor

from data.tensors.source import TensorSource


class SafeTensorsSource:
    """Safetensors tensor source

    Lazily loads tensors from safetensors files, caching the loaded tensor
    and length to avoid repeated file opens. Thread-safe for DataLoader usage.
    """

    def __init__(self, *, path: Path, tensor_name: str) -> None:
        """Initialize safetensors source

        Stores the file path and tensor name, deferring actual loading until
        first access to avoid opening files unnecessarily.
        """
        self.path = path
        self.tensor_name = tensor_name
        self._length: int | None = None
        self._tensor: Tensor | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Get number of samples

        Opens the safetensors file on first call to read tensor shape, then
        caches the length to avoid repeated file opens. Thread-safe through
        double-checked locking.
        """
        cached = self._length
        if cached is not None:
            return int(cached)
        with self._lock:
            cached2 = self._length
            if cached2 is not None:
                return int(cached2)
            from safetensors import safe_open

            with safe_open(str(self.path), framework="pt", device="cpu") as f:
                t = f.get_tensor(self.tensor_name)
                self._length = int(t.shape[0])
                return int(self._length)

    def get(self, idx: int) -> Tensor:
        """Get tensor at index

        Loads the entire tensor from safetensors on first access, then caches
        it in memory. For very large tensors, consider using mmap-based access
        instead of full loading.
        """
        t = self._tensor
        if t is None:
            with self._lock:
                t = self._tensor
                if t is None:
                    from safetensors import safe_open

                    with safe_open(str(self.path), framework="pt", device="cpu") as f:
                        t = f.get_tensor(self.tensor_name)
                    self._tensor = t
                    if self._length is None:
                        self._length = int(t.shape[0])
        return t[int(idx)]
