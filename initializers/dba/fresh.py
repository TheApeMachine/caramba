"""DBA fresh initializer for full attention replacement.

This initializer implements the "routing hypothesis" approach: attention is primarily
a routing mechanism, so when replacing standard attention with DBA, we initialize
all DBA projections fresh (Xavier uniform) rather than trying to preserve teacher
patterns via SVD.

The pretrained FFN layers and embeddings already contain the model's knowledge.
The fresh DBA layer just needs to learn how to route information through them.

Use Case:
    - Full attention replacement experiments
    - Testing the routing hypothesis
    - When you want DBA to learn its own semantic/geometric patterns from scratch

Configuration:
    Set `dba_init: fresh` in the manifest's train config.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from caramba.carmath.sketch import stable_int_hash
from caramba.initializers.dba.base import DBAInitializer


class DBAFresh(DBAInitializer):
    """Complete random initialization for DBA - ignores teacher weights entirely.

    Unlike DBASVD (which decomposes teacher weights) or DBARandom (which still
    references teacher shape), this initializer treats the DBA layer as genuinely
    new. All projections (Q_sem, K_sem, Q_geo, K_geo) receive Xavier uniform
    initialization based solely on their own dimensions.

    This is the correct initializer for testing the routing hypothesis: the model
    already knows things via FFN/embeddings, and we're teaching it a new way to
    route that knowledge through attention.
    """

    def initialize(
        self,
        *,
        sem_weight: Tensor,
        geo_weight: Tensor,
        teacher_weight: Tensor,  # Ignored - kept for interface compatibility
        sem_dim: int,
        geo_dim: int,
        seed: str,
    ) -> None:
        """Initialize semantic and geometric projections with Xavier uniform.

        Args:
            sem_weight: Semantic projection weight tensor to initialize (out_features, in_features)
            geo_weight: Geometric projection weight tensor to initialize (out_features, in_features)
            teacher_weight: IGNORED - present only for interface compatibility
            sem_dim: Total semantic dimension (used for bound calculation)
            geo_dim: Total geometric dimension (used for bound calculation)
            seed: Deterministic seed string for reproducibility
        """
        d_in = int(sem_weight.shape[1])  # d_model

        # Xavier uniform for semantic projection
        # bound = sqrt(6 / (fan_in + fan_out))
        gen_sem = torch.Generator(device=sem_weight.device)
        gen_sem.manual_seed(stable_int_hash(f"{seed}:sem:fresh"))
        bound_sem = math.sqrt(6.0 / (d_in + sem_dim))
        sem_weight.data.uniform_(-bound_sem, bound_sem, generator=gen_sem)

        # Xavier uniform for geometric projection
        gen_geo = torch.Generator(device=geo_weight.device)
        gen_geo.manual_seed(stable_int_hash(f"{seed}:geo:fresh"))
        bound_geo = math.sqrt(6.0 / (d_in + geo_dim))
        geo_weight.data.uniform_(-bound_geo, bound_geo, generator=gen_geo)


def init_fresh_linear(
    linear: nn.Linear,
    *,
    seed: str,
    suffix: str = "",
    scale: float = 1.0,
) -> None:
    """Xavier uniform initialization for a linear layer.

    Helper for initializing V and O projections when doing full replacement.

    Args:
        linear: The nn.Linear module to initialize
        seed: Deterministic seed string
        suffix: Additional suffix for seed uniqueness
        scale: Scale factor for initialization (use small values like 0.02 for output projections
               that feed into residual streams to avoid disrupting pretrained norms)
    """
    d_in = int(linear.in_features)
    d_out = int(linear.out_features)

    gen = torch.Generator(device=linear.weight.device)
    gen.manual_seed(stable_int_hash(f"{seed}:{suffix}:fresh"))

    # Xavier uniform base, then scale down
    bound = math.sqrt(6.0 / (d_in + d_out)) * scale
    linear.weight.data.uniform_(-bound, bound, generator=gen)

    if linear.bias is not None:
        linear.bias.data.zero_()
