from __future__ import annotations

import unittest

from research.dba.behavioral_suite_v2.scoring import MatchQuality, classify_match


class TestBehavioralV2Scoring(unittest.TestCase):
    def test_int_exact(self) -> None:
        q, _flags, _fl = classify_match("4", "4", prompt="")
        self.assertEqual(q, MatchQuality.EXACT)

    def test_int_soft_equation(self) -> None:
        q, _flags, _fl = classify_match("2 + 2 = 4", "4", prompt="")
        self.assertEqual(q, MatchQuality.PARTIAL)

    def test_int_reject_number_list(self) -> None:
        q, _flags, _fl = classify_match("1 2 3 4 5 6 7 8", "8", prompt="")
        self.assertEqual(q, MatchQuality.NONE)


if __name__ == "__main__":
    unittest.main()

