"""
svd_math_test validates the DBA SVD initializer math.

These tests are deliberately small and CPU-only so they can be run frequently.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

# Allow running under an installed `caramba` package or directly from a repo checkout.
try:
    from caramba.initializers.dba.svd import DBASVD  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # File lives at: <repo_root>/initializers/dba/svd_math_test.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from initializers.dba.svd import DBASVD


class DBASVDMathTest(unittest.TestCase):
    def test_write_weights_matches_diagS_times_Vh_prefix(self) -> None:
        torch.manual_seed(0)
        d_in = 16
        d_out = 12
        sem_dim = 5
        geo_dim = 4
        rank = sem_dim + geo_dim

        teacher_weight = torch.randn(d_out, d_in, dtype=torch.float32)
        sem_w = torch.empty(sem_dim, d_in, dtype=torch.float32)
        geo_w = torch.empty(geo_dim, d_in, dtype=torch.float32)

        init = DBASVD()
        init.initialize(
            sem_weight=sem_w,
            geo_weight=geo_w,
            teacher_weight=teacher_weight,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            seed="unit",
        )

        # Compute reference Wr = diag(S_r) @ Vh_r using full SVD (deterministic).
        _U, S, Vh = torch.linalg.svd(teacher_weight, full_matrices=False)
        Wr = (S[:rank].view(-1, 1) * Vh[:rank, :]).contiguous()

        self.assertTrue(torch.allclose(sem_w, Wr[:sem_dim, :], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(geo_w, Wr[sem_dim : sem_dim + geo_dim, :], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.isfinite(sem_w).all())
        self.assertTrue(torch.isfinite(geo_w).all())

    def test_full_rank_path_is_finite(self) -> None:
        torch.manual_seed(0)
        d_in = 8
        d_out = 8
        # rank >= full_rank triggers torch.linalg.svd path in computeTruncatedSvd
        sem_dim = 6
        geo_dim = 6
        teacher_weight = torch.randn(d_out, d_in, dtype=torch.float32)

        sem_w = torch.empty(sem_dim, d_in, dtype=torch.float16)
        geo_w = torch.empty(geo_dim, d_in, dtype=torch.float16)

        init = DBASVD()
        init.initialize(
            sem_weight=sem_w,
            geo_weight=geo_w,
            teacher_weight=teacher_weight,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            seed="full",
        )

        self.assertTrue(torch.isfinite(sem_w).all())
        self.assertTrue(torch.isfinite(geo_w).all())


if __name__ == "__main__":
    unittest.main()

