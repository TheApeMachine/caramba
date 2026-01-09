"""Resonant cleanup (alias).

Preferred import path: `resonant.core.resonant_cleanup`.
Implementation currently lives in `synthnn.core.resonant_cleanup`.
"""

from synthnn.core.resonant_cleanup import (
    ResonantCleanupConfig,
    cleanup_resonant_only,
    detect_artifacts_resonant,
)

__all__ = [
    "ResonantCleanupConfig",
    "cleanup_resonant_only",
    "detect_artifacts_resonant",
]

