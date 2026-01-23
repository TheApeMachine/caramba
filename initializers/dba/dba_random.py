"""DBA random initializer.

NOTE: This file is intentionally *not* named `random.py`.
When `initializers/dba/svd_math_test.py` (or any module in this directory) is
executed as a script, Python puts this directory on `sys.path[0]`. A local
`random.py` would then shadow the stdlib `random` module, which breaks imports
inside PyTorch (via `tempfile -> random`).
"""

from __future__ import annotations

import torch
from torch import Tensor

from carmath.sketch import stable_int_hash
from initializers.dba.base import DBAInitializer


class DBARandom(DBAInitializer):
    """Deterministic random DBA initializer.

    Initializes semantic and geometric projections with independent random
    matrices drawn from a unit normal distribution and scaled by 1/sqrt(d_in),
    matching common transformer initialization conventions.
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
        d_in = int(teacher_weight.shape[1])
        if d_in <= 0:
            raise ValueError(f"Invalid teacher_weight shape for random init: {tuple(teacher_weight.shape)}")

        gen = torch.Generator(device=sem_weight.device)
        gen.manual_seed(int(stable_int_hash(str(seed))))

        scale = float(d_in) ** -0.5
        sem_weight.data.normal_(mean=0.0, std=scale, generator=gen)

        gen2 = torch.Generator(device=geo_weight.device)
        gen2.manual_seed(int(stable_int_hash(str(seed) + ":geo")))
        geo_weight.data.normal_(mean=0.0, std=scale, generator=gen2)

