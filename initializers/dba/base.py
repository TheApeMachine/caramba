"""DBA initializer base.

Defines the initialization contract for decoupled attention (DBA) projections.
This keeps upcycle loaders focused on mapping weights while initializers own the
math policy of how to seed new parameters.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class DBAInitializer(Protocol):
    """Initializer contract for DBA projections.

    Implementations initialize target semantic/geometric projection weights from
    a teacher projection weight matrix.
    """

    def initialize(
        self,
        *,
        sem_weight: Tensor,
        geo_weight: Tensor,
        teacher_weight: Tensor,
        sem_dim: int,
        geo_dim: int,
        seed: str,
    ) -> None:
        """Initialize semantic and geometric weights in-place."""
        ...

