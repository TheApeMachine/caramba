"""DBA initializers.

DBA (decoupled attention) uses separate semantic and geometric projections. When
upcycling from a standard attention checkpoint, we need a deterministic policy
for initializing these projections from teacher weights.

Available initializers:
    - DBASVD: Initialize from SVD decomposition of teacher weights (preserves patterns)
    - DBARandom: Random init for Q/K semantic/geometric, copy V/O from teacher
    - DBAFresh: Complete random init for ALL projections (routing hypothesis)
"""

from __future__ import annotations

from initializers.dba.base import DBAInitializer
from initializers.dba.fresh import DBAFresh
from initializers.dba.dba_random import DBARandom
from initializers.dba.svd import DBASVD

__all__ = ["DBAInitializer", "DBASVD", "DBARandom", "DBAFresh"]

