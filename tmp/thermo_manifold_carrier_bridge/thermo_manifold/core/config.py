from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    """Simulation configuration.

    Notes:
    - `dt` is the integration step (a physical simulation timescale, not a tuned ML hyperparameter).
    - `eps` is numerical safety for division/log.
    """

    dt: float = 1e-2
    eps: float = 1e-8
