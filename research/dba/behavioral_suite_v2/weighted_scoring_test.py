from __future__ import annotations

import unittest

from research.dba.behavioral_suite_v2.weighted_scoring import MatchType, classify_match_type


class TestWeightedScoringMatchType(unittest.TestCase):
    def test_numeric_list_is_not_contained(self) -> None:
        # Reject degenerate counting/list outputs.
        self.assertEqual(classify_match_type("1 2 3 4 5 6 7 8", "8"), MatchType.NONE)

    def test_numeric_equation_is_contained(self) -> None:
        self.assertEqual(classify_match_type("2 + 2 = 4", "4"), MatchType.CONTAINED)

    def test_numeric_answer_statement_is_contained(self) -> None:
        self.assertEqual(classify_match_type("the answer is 8", "8"), MatchType.CONTAINED)


if __name__ == "__main__":
    unittest.main()

