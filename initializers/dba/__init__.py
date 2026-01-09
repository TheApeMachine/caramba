"""DBA initializers.

DBA (decoupled attention) uses separate semantic and geometric projections. When
upcycling from a standard attention checkpoint, we need a deterministic policy
for initializing these projections from teacher weights.
"""

from __future__ import annotations

from caramba.initializers.dba.base import DBAInitializer
from caramba.initializers.dba.random import DBARandom
from caramba.initializers.dba.svd import DBASVD

__all__ = ["DBAInitializer", "DBASVD", "DBARandom"]

