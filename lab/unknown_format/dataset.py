"""Unknown format lab dataset

Emits reproducible raw byte samples with ground truth and auto-generated tests.
This is intended for CCP demo loops and (later) model training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from lab.unknown_format.format_family import FormatFamily, FormatSpec
from lab.unknown_format.oracle import FormatOracle, Record
from lab.unknown_format.test_gen import ToolTestGenerator


@dataclass(frozen=True, slots=True)
class UnknownFormatSample:
    """One lab sample."""

    spec: FormatSpec
    raw: bytes
    records: list[Record]
    tests: str

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": {"version": int(self.spec.version), "record_count": int(self.spec.record_count)},
            "raw_hex": self.raw.hex(),
            "records": [r.to_json() for r in self.records],
            "tests": self.tests,
        }


class UnknownFormatLabDataset:
    """Unknown format lab dataset.

    This is intentionally torch-free. It is a generator of reproducible samples.
    """

    def __init__(self, *, seed: int, n_items: int) -> None:
        self.seed = int(seed)
        self.n_items = int(n_items)
        if self.n_items < 1:
            raise ValueError("n_items must be >= 1")
        self.family = FormatFamily(seed=self.seed)
        self.oracle = FormatOracle()
        self.tests = ToolTestGenerator()

    def __len__(self) -> int:
        return int(self.n_items)

    def sample(self, index: int) -> UnknownFormatSample:
        """Generate a deterministic sample for an index."""
        if int(index) < 0 or int(index) >= int(self.n_items):
            raise IndexError("index out of range")
        rng = random.Random(self.seed + int(index))
        spec = self.family.sample_spec(nonce=int(index))
        records = [Record(payload=bytes([rng.randrange(0, 256) for _ in range(rng.randrange(1, 8))])) for _ in range(spec.record_count)]
        raw = self.oracle.encode(spec=spec, records=records)
        expected = [r.payload.hex() for r in records]
        tests_src = self.tests.build_tests(sample_bytes_hex=raw.hex(), expected_payload_hex=expected)
        return UnknownFormatSample(spec=spec, raw=raw, records=records, tests=tests_src)

