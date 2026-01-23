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

    def test_boolean_only_first_token_counts(self) -> None:
        self.assertEqual(classify_match_type("true, false, true", "false"), MatchType.NONE)
        self.assertEqual(classify_match_type("false, true, false", "false"), MatchType.CONTAINED)
        self.assertEqual(classify_match_type("yes no yes", "no"), MatchType.NONE)
        self.assertEqual(classify_match_type("no yes", "no"), MatchType.CONTAINED)

    def test_multiline_mapping_only_first_line_counts(self) -> None:
        # Few-shot mapping style: don't allow "spraying" multiple candidates.
        expected = "down up"
        baseline_out = (
            "up down ->\n"
            "down down ->\n"
            "down up ->\n"
            "up down ->\n"
        )
        gated_out = "down up down\nup up up\n"
        self.assertEqual(classify_match_type(baseline_out, expected), MatchType.NONE)
        self.assertEqual(classify_match_type(gated_out, expected), MatchType.CONTAINED)

    def test_presence_contained_can_be_enabled_per_test(self) -> None:
        # When explicitly enabled, allow CONTAINED if expected appears anywhere in output,
        # but only if the expected answer is not already present verbatim in the prompt.
        prompt = "cat -> the cat\ndog -> the dog\nbird ->"
        expected = "the bird"
        output = (
            "cat -> the cat\n"
            "dog -> the dog\n"
            "bird -> the bird\n"
        )

        # Default behavior: multi-line mapping only checks the first line -> NONE.
        self.assertEqual(classify_match_type(output, expected), MatchType.NONE)

        # Enabled: expected appears later and is not in prompt -> CONTAINED.
        self.assertEqual(
            classify_match_type(
                output,
                expected,
                prompt=prompt,
                allow_presence_contained=True,
            ),
            MatchType.CONTAINED,
        )

    def test_presence_contained_does_not_allow_prompt_copy(self) -> None:
        prompt = "the bird"
        expected = "the bird"
        output = "some junk\n... the bird ...\nmore junk\n"
        # Even when enabled, if the expected is already in the prompt verbatim,
        # do not award presence-based CONTAINED.
        self.assertEqual(
            classify_match_type(
                output,
                expected,
                prompt=prompt,
                allow_presence_contained=True,
            ),
            MatchType.NONE,
        )


if __name__ == "__main__":
    unittest.main()

