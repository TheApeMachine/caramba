from __future__ import annotations

import unittest

from lab.unknown_format.dataset import UnknownFormatLabDataset
from lab.unknown_format.oracle import FormatOracle


class UnknownFormatLabTest(unittest.TestCase):
    def test_sample_is_deterministic(self) -> None:
        ds = UnknownFormatLabDataset(seed=1, n_items=3)
        a = ds.sample(0).to_json()
        b = ds.sample(0).to_json()
        self.assertEqual(a["raw_hex"], b["raw_hex"])

    def test_oracle_round_trip(self) -> None:
        ds = UnknownFormatLabDataset(seed=2, n_items=1)
        s = ds.sample(0)
        oracle = FormatOracle()
        rec = oracle.decode(spec=s.spec, raw=s.raw)
        self.assertEqual([r.payload.hex() for r in rec], [r.payload.hex() for r in s.records])

