"""Resonant node

Represents a single complex-valued oscillator. The node's state is stored as a
single complex number (signal), which encodes both amplitude and phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ResonantNode:
    """Resonant node

    Used as the fundamental element of a resonant network. Each node evolves by:
    - rotating at its natural frequency
    - receiving complex coupling inputs
    - applying damping (energy dissipation)
    """

    node_id: str
    natural_freq: float = 1.0
    damping: float = 0.1
    signal: complex = 1.0 + 0.0j
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.natural_freq = float(self.natural_freq)
        damping = float(self.damping)
        if not np.isfinite(damping):
            damping = 0.0
        # Treat damping as a non-negative decay rate (not a [0,1] fraction). This
        # is more useful for signal-processing style resonant filters where the
        # desired time constant can be much shorter than 1s.
        self.damping = max(0.0, damping)
        self.signal = complex(self.signal)

    def amplitude(self) -> float:
        """Amplitude of the oscillator."""

        return float(abs(self.signal))

    def phase(self) -> float:
        """Phase angle in radians."""

        return float(np.angle(self.signal))

    def setPolar(self, *, amplitude: float, phase: float) -> None:
        """Set the state from amplitude/phase."""

        amp = max(0.0, float(amplitude))
        ph = float(phase) % (2.0 * float(np.pi))
        self.signal = amp * np.exp(1j * ph)

    def step(self, *, dt: float, coupling: complex = 0.0 + 0.0j, damping_override: float | None = None) -> None:
        """Advance node dynamics by dt."""

        dt = float(dt)
        damping = self.damping
        if damping_override is not None:
            try:
                damping = float(damping_override)
            except (TypeError, ValueError):
                damping = float(self.damping)
            if not np.isfinite(damping):
                damping = float(self.damping)
            damping = max(0.0, float(damping))

        # Phase advance, then coupling, then damping.
        self.signal *= np.exp(1j * 2.0 * np.pi * self.natural_freq * dt)
        self.signal += coupling
        decay = float(np.exp(-damping * dt))
        self.signal *= decay

    def energy(self) -> float:
        """Instantaneous energy proxy."""

        a = self.amplitude()
        return float(0.5 * a * a)

