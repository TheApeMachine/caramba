"""Format family

Defines a parameterized family of simple binary record formats.
The family is seeded and deterministic so datasets and tests are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Format specification.

    v1 uses:
    - version byte
    - record_count byte
    - repeated records: [len:u8][payload bytes][checksum:u8]
    """

    version: int
    record_count: int

    def validate(self) -> None:
        if int(self.version) < 0 or int(self.version) > 255:
            raise ValueError("version must be u8")
        if int(self.record_count) < 0 or int(self.record_count) > 255:
            raise ValueError("record_count must be u8")


class FormatFamily:
    """Unknown format family generator."""

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)

    def sample_spec(self, *, nonce: int) -> FormatSpec:
        """Sample a deterministic format spec for a nonce."""
        rng = random.Random(int(self.seed) + int(nonce))
        version = rng.randrange(1, 4)
        record_count = rng.randrange(1, 6)
        spec = FormatSpec(version=version, record_count=record_count)
        spec.validate()
        return spec

