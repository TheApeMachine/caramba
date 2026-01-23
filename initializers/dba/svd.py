"""DBA SVD initializer.

Uses a truncated SVD of the teacher projection to seed semantic and geometric
projection weights. This makes upcycling deterministic and meaningfully aligned
to the teacher's strongest directions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from carmath import randomized_svd
from initializers.dba.base import DBAInitializer


class DBASVD(DBAInitializer):
    """Initialize DBA projections from teacher weights using SVD.

    The teacher projection is A = teacher_weight (d_out x d_in), with SVD:

      A = U S V^T

    We build a compressed representation in the singular output basis:

      W_comp = diag(S_r) V_r^T  (r x d_in)

    and write the first sem_dim rows into sem_weight and the next geo_dim rows
    into geo_weight, yielding disjoint components.
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
        A = self.teacherMatrix(teacher_weight=teacher_weight, device=sem_weight.device)
        target_rank = self.targetRank(sem_dim=int(sem_dim), geo_dim=int(geo_dim))
        S, Vh = self.computeTruncatedSvd(A=A, rank=target_rank, seed=str(seed))
        Wr = self.compressedWeights(S=S, Vh=Vh, rank=int(target_rank))
        self.writeWeights(
            sem_weight=sem_weight,
            geo_weight=geo_weight,
            Wr=Wr,
            sem_dim=int(sem_dim),
            geo_dim=int(geo_dim),
        )

    def teacherMatrix(self, *, teacher_weight: Tensor, device: torch.device) -> Tensor:
        """Return the teacher projection as float32 on `device`."""

        return teacher_weight.to(device=device, dtype=torch.float32)

    def targetRank(self, *, sem_dim: int, geo_dim: int) -> int:
        """Return the target truncated rank for DBA."""

        target_rank = int(sem_dim) + int(geo_dim)
        if target_rank <= 0:
            raise ValueError(
                f"Invalid DBA target_rank={target_rank} (sem_dim={sem_dim}, geo_dim={geo_dim})"
            )
        return int(target_rank)

    def computeTruncatedSvd(self, *, A: Tensor, rank: int, seed: str) -> tuple[Tensor, Tensor]:
        """Compute truncated SVD components needed for compression."""

        full_rank = min(int(A.shape[0]), int(A.shape[1]))
        if int(rank) >= full_rank:
            _U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return S, Vh

        # For small matrices, prefer the exact, deterministic SVD.
        #
        # This keeps unit tests stable and avoids approximation error for the
        # common upcycling case where projections are relatively small.
        #
        # For large projections (e.g. 4096x4096), full SVD is too expensive, so we
        # fall back to a seeded randomized SVD.
        if full_rank <= 1024 and int(A.numel()) <= 2_000_000:
            _U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return S, Vh

        _U, S, Vh = randomized_svd(A, rank=int(rank), n_iter=2, oversample=8, seed=str(seed))
        return S, Vh

    def compressedWeights(self, *, S: Tensor, Vh: Tensor, rank: int) -> Tensor:
        """Build the compressed weight matrix diag(S) @ Vh."""

        r = min(int(S.size(0)), int(rank))
        return (S[:r].view(-1, 1) * Vh[:r, :]).contiguous()

    def writeWeights(
        self,
        *,
        sem_weight: Tensor,
        geo_weight: Tensor,
        Wr: Tensor,
        sem_dim: int,
        geo_dim: int,
    ) -> None:
        """Write semantic and geometric weights into the target tensors."""

        sem_weight.data.zero_()
        geo_weight.data.zero_()
        self.writeSemantic(sem_weight=sem_weight, Wr=Wr, sem_dim=int(sem_dim))
        self.writeGeometric(geo_weight=geo_weight, Wr=Wr, sem_dim=int(sem_dim), geo_dim=int(geo_dim))

    def writeSemantic(self, *, sem_weight: Tensor, Wr: Tensor, sem_dim: int) -> None:
        """Write semantic rows from Wr into sem_weight."""

        sem_rows = min(int(sem_dim), int(Wr.size(0)))
        if sem_rows <= 0:
            return
        sem_weight.data[:sem_rows, :].copy_(Wr[:sem_rows, :].to(dtype=sem_weight.dtype))

    def writeGeometric(
        self, *, geo_weight: Tensor, Wr: Tensor, sem_dim: int, geo_dim: int
    ) -> None:
        """Write geometric rows from Wr into geo_weight."""

        geo_start = int(sem_dim)
        geo_end = int(sem_dim) + int(geo_dim)
        rank = int(Wr.size(0))
        geo_rows = max(0, min(rank, geo_end) - geo_start)
        if geo_rows <= 0:
            return
        geo_weight.data[:geo_rows, :].copy_(
            Wr[geo_start : geo_start + geo_rows, :].to(dtype=geo_weight.dtype)
        )

