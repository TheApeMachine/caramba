"""Phasor math utilities

Provides operations for converting values to unit phasors, projecting vectors
onto the unit circle, and measuring phase deltas for convergence detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PhasorMath:
    """Phasor math helpers

    These utilities are kept as methods (not loose functions) so they can be
    composed and injected into higher-level systems without relying on globals.
    """

    projection_eps_c64: float = 1e-7
    projection_eps_c128: float = 1e-12

    def as2d(self, x: np.ndarray) -> np.ndarray:
        """Ensure patterns are 2D."""

        x = np.asarray(x)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim != 2:
            raise ValueError("patterns must be 1D or 2D array-like")
        return x

    def toPhasors(self, x: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        """Convert angles or complex values to unit phasors."""

        x = np.asarray(x)
        if np.iscomplexobj(x):
            return self.normalizeComplex(x=x, dtype=dtype)
        angles = x.astype(np.float64, copy=False)
        out = np.exp(1j * angles)
        return out.astype(dtype, copy=False)

    def normalizeComplex(self, *, x: np.ndarray, dtype: np.dtype, eps: float = 1e-12) -> np.ndarray:
        """Normalize complex values to unit magnitude (zeros become 1+0j)."""

        x = np.asarray(x)
        mag = np.abs(x)
        out = np.empty_like(x, dtype=dtype)
        nonzero = mag > float(eps)
        out[nonzero] = (x[nonzero] / mag[nonzero]).astype(dtype, copy=False)
        out[~nonzero] = 1.0 + 0.0j
        return out

    def projectUnitOrZero(self, x: np.ndarray, *, eps: float, dtype: np.dtype) -> np.ndarray:
        """Project to unit circle elementwise, preserving zeros as unknowns."""

        x = np.asarray(x)
        mag = np.abs(x)
        out = np.zeros_like(x, dtype=dtype)
        nonzero = mag > float(eps)
        out[nonzero] = (x[nonzero] / mag[nonzero]).astype(dtype, copy=False)
        return out

    def meanPhaseDelta(self, *, a: np.ndarray, b: np.ndarray, eps: float) -> float:
        """Mean absolute phase delta between vectors, ignoring unknowns."""

        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        valid = (np.abs(a) > float(eps)) & (np.abs(b) > float(eps))
        if not np.any(valid):
            return 0.0
        d = np.angle(a[valid] * np.conj(b[valid]))
        return float(np.mean(np.abs(d)))

    def defaultProjectionEps(self, *, dtype: np.dtype) -> float:
        """Choose a practical eps based on complex dtype."""

        dtype = np.dtype(dtype)
        if dtype == np.dtype(np.complex64):
            return float(self.projection_eps_c64)
        return float(self.projection_eps_c128)

