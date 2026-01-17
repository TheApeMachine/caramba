"""
Weighted scoring system with hard/soft scores and difficulty weighting.

This module provides a scoring system that:
1. Classifies matches as EXACT (1.0), CONTAINED (0.5), or NONE (0.0)
2. Weights scores based on baseline/teacher difficulty
3. Aggregates results with per-category and per-difficulty breakdowns

Match Types:
- EXACT: Output exactly matches expected (after stripping whitespace)
- CONTAINED: Expected answer appears within output (with prefix/suffix tokens)
- NONE: Expected answer not found in output

Difficulty Weighting (based on baseline performance):
- Baseline EXACT (easy): 1.0x weight
- Baseline CONTAINED (medium): 2.0x weight
- Baseline NONE (hard): 3.0x weight

Example:
    Expected: "4", Output: "4" → EXACT (1.0 points)
    Expected: "4", Output: "2 + 2 = 4" → CONTAINED (0.5 points)
    Expected: "4", Output: "5" → NONE (0.0 points)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MatchType(IntEnum):
    """Match type for output vs expected."""
    NONE = 0       # Expected not found in output
    CONTAINED = 1  # Expected found within output (prefix/suffix allowed)
    EXACT = 2      # Exact match


# Raw scores for each match type
MATCH_SCORES: dict[MatchType, float] = {
    MatchType.NONE: 0.0,
    MatchType.CONTAINED: 0.5,
    MatchType.EXACT: 1.0,
}

# Difficulty weights based on baseline performance
DIFFICULTY_WEIGHTS: dict[MatchType, float] = {
    MatchType.NONE: 3.0,      # Baseline wrong → hard problem
    MatchType.CONTAINED: 2.0,  # Baseline had extra tokens → medium
    MatchType.EXACT: 1.0,      # Baseline perfect → easy
}

# Human-readable difficulty names
DIFFICULTY_NAMES: dict[float, str] = {
    1.0: "easy",
    2.0: "medium",
    3.0: "hard",
}


def classify_match_type(output: str, expected: str) -> MatchType:
    """
    Classify match type between output and expected.

    Args:
        output: Model's output string
        expected: Expected answer string

    Returns:
        MatchType.EXACT if output == expected (stripped)
        MatchType.CONTAINED if expected appears anywhere in output
        MatchType.NONE otherwise
    """
    output_clean = output.strip()
    expected_clean = str(expected).strip()

    if not expected_clean:
        return MatchType.NONE

    # EXACT: Perfect match (full output)
    if output_clean == expected_clean:
        return MatchType.EXACT

    # EXACT: First line matches expected exactly
    first_line = output_clean.split('\n')[0].strip()
    if first_line == expected_clean:
        return MatchType.EXACT

    # CONTAINED: Expected appears anywhere in output
    # For short expected values (<=3 chars), require word boundary to avoid false positives
    if len(expected_clean) <= 3:
        # Check word boundaries - expected should appear as isolated token
        pattern = r'(?:^|[\s\[\(\{,;:])' + re.escape(expected_clean) + r'(?:[\s\]\)\},;:.]|$)'
        if re.search(pattern, output_clean):
            return MatchType.CONTAINED
        # Also check if it's in the word list of first line
        words = first_line.split()
        if expected_clean in words:
            return MatchType.CONTAINED
        # Check if first line starts with expected (common for numeric answers)
        if first_line.startswith(expected_clean) and (
            len(first_line) == len(expected_clean) or
            not first_line[len(expected_clean)].isalnum()
        ):
            return MatchType.CONTAINED
    else:
        # For longer expected values, simple containment check
        if expected_clean in output_clean:
            return MatchType.CONTAINED

    return MatchType.NONE


@dataclass
class WeightedTestScore:
    """Score for a single test with difficulty weighting."""
    test_id: str
    model_name: str
    baseline_name: str
    category: str = ""

    # Match types
    model_match: MatchType = MatchType.NONE
    baseline_match: MatchType = MatchType.NONE

    # Scores
    raw_score: float = 0.0           # 0.0, 0.5, or 1.0
    difficulty_weight: float = 1.0   # 1.0, 2.0, or 3.0
    weighted_score: float = 0.0      # raw_score × difficulty_weight

    # Max possible weighted score for this test
    max_weighted_score: float = 1.0  # 1.0 × difficulty_weight

    # Raw data for debugging
    expected: str = ""
    model_output: str = ""
    baseline_output: str = ""

    @property
    def difficulty_name(self) -> str:
        """Human-readable difficulty name."""
        return DIFFICULTY_NAMES.get(self.difficulty_weight, "unknown")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "test_id": self.test_id,
            "model_name": self.model_name,
            "baseline_name": self.baseline_name,
            "category": self.category,
            "model_match": self.model_match.name,
            "baseline_match": self.baseline_match.name,
            "raw_score": self.raw_score,
            "difficulty_weight": self.difficulty_weight,
            "difficulty_name": self.difficulty_name,
            "weighted_score": self.weighted_score,
            "max_weighted_score": self.max_weighted_score,
            "expected": self.expected,
            "model_output": self.model_output[:200],  # Truncate for readability
            "baseline_output": self.baseline_output[:200],
        }


@dataclass
class DifficultyBreakdown:
    """Breakdown of scores for a single difficulty tier."""
    difficulty: str  # "easy", "medium", "hard"
    weight: float    # 1.0, 2.0, 3.0
    count: int = 0
    exact_count: int = 0
    contained_count: int = 0
    none_count: int = 0
    weighted_score: float = 0.0
    weighted_max: float = 0.0

    @property
    def accuracy(self) -> float:
        """Weighted accuracy for this difficulty tier."""
        return self.weighted_score / self.weighted_max if self.weighted_max > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "difficulty": self.difficulty,
            "weight": self.weight,
            "count": self.count,
            "exact_count": self.exact_count,
            "contained_count": self.contained_count,
            "none_count": self.none_count,
            "weighted_score": self.weighted_score,
            "weighted_max": self.weighted_max,
            "accuracy": self.accuracy,
        }


@dataclass
class CategoryBreakdown:
    """Breakdown of scores for a single category."""
    category: str
    count: int = 0
    exact_count: int = 0
    contained_count: int = 0
    none_count: int = 0
    raw_score: float = 0.0
    weighted_score: float = 0.0
    weighted_max: float = 0.0

    @property
    def hard_accuracy(self) -> float:
        """EXACT match rate."""
        return self.exact_count / self.count if self.count > 0 else 0.0

    @property
    def soft_accuracy(self) -> float:
        """EXACT + CONTAINED match rate."""
        return (self.exact_count + self.contained_count) / self.count if self.count > 0 else 0.0

    @property
    def weighted_accuracy(self) -> float:
        """Weighted accuracy."""
        return self.weighted_score / self.weighted_max if self.weighted_max > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category": self.category,
            "count": self.count,
            "exact_count": self.exact_count,
            "contained_count": self.contained_count,
            "none_count": self.none_count,
            "raw_score": self.raw_score,
            "weighted_score": self.weighted_score,
            "weighted_max": self.weighted_max,
            "hard_accuracy": self.hard_accuracy,
            "soft_accuracy": self.soft_accuracy,
            "weighted_accuracy": self.weighted_accuracy,
        }


@dataclass
class WeightedModelSummary:
    """Aggregated weighted scores for a model."""
    model_name: str
    baseline_name: str

    # Test counts
    total_tests: int = 0
    exact_count: int = 0
    contained_count: int = 0
    none_count: int = 0

    # Raw scores (unweighted)
    raw_score_sum: float = 0.0
    raw_score_max: float = 0.0  # = total_tests × 1.0
    hard_accuracy: float = 0.0  # exact_count / total_tests
    soft_accuracy: float = 0.0  # (exact + contained) / total_tests

    # Weighted scores
    weighted_score_sum: float = 0.0
    weighted_score_max: float = 0.0  # sum of max_weighted_score per test
    weighted_accuracy: float = 0.0   # weighted_score_sum / weighted_score_max

    # Breakdown by baseline difficulty
    by_difficulty: dict[str, DifficultyBreakdown] = field(default_factory=dict)

    # Per-category breakdown
    by_category: dict[str, CategoryBreakdown] = field(default_factory=dict)

    # Per-test scores (for detailed analysis)
    test_scores: list[WeightedTestScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "baseline_name": self.baseline_name,
            "total_tests": self.total_tests,
            "exact_count": self.exact_count,
            "contained_count": self.contained_count,
            "none_count": self.none_count,
            "raw_score_sum": self.raw_score_sum,
            "raw_score_max": self.raw_score_max,
            "hard_accuracy": self.hard_accuracy,
            "soft_accuracy": self.soft_accuracy,
            "weighted_score_sum": self.weighted_score_sum,
            "weighted_score_max": self.weighted_score_max,
            "weighted_accuracy": self.weighted_accuracy,
            "by_difficulty": {k: v.to_dict() for k, v in self.by_difficulty.items()},
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
        }


class WeightedScorer:
    """
    Compute weighted scores relative to a baseline model.

    Usage:
        scorer = WeightedScorer(baseline_name="baseline")

        # Register tests
        for test in tests:
            scorer.add_test(test.id, test.expected, test.category)

        # Add outputs for all models
        for model_name, outputs in model_outputs.items():
            for test_id, output in outputs.items():
                scorer.add_output(test_id, model_name, output)

        # Get summaries
        for model_name in ["sem16", "sem8"]:
            summary = scorer.get_model_summary(model_name)
            print(f"{model_name}: weighted={summary.weighted_accuracy:.2%}")
    """

    def __init__(self, baseline_name: str):
        self.baseline_name = baseline_name
        self.tests: dict[str, dict[str, Any]] = {}  # {test_id: {expected, category}}
        self.model_outputs: dict[str, dict[str, str]] = {}  # {model: {test_id: output}}
        self.baseline_matches: dict[str, MatchType] = {}  # {test_id: match_type}

    def add_test(self, test_id: str, expected: str, category: str = "") -> None:
        """Register a test case."""
        self.tests[test_id] = {
            "expected": str(expected),
            "category": category,
        }

    def add_output(self, test_id: str, model_name: str, output: str) -> None:
        """Add model output for a test."""
        if test_id not in self.tests:
            raise KeyError(f"Test {test_id} not registered. Call add_test first.")

        if model_name not in self.model_outputs:
            self.model_outputs[model_name] = {}
        self.model_outputs[model_name][test_id] = output

        # If this is baseline, compute match type
        if model_name == self.baseline_name:
            expected = self.tests[test_id]["expected"]
            self.baseline_matches[test_id] = classify_match_type(output, expected)

    def compute_weighted_scores(self, model_name: str) -> list[WeightedTestScore]:
        """Compute weighted scores for a model vs baseline."""
        scores = []

        for test_id, test_info in self.tests.items():
            expected = test_info["expected"]
            category = test_info.get("category", "")

            model_output = self.model_outputs.get(model_name, {}).get(test_id, "")
            baseline_output = self.model_outputs.get(self.baseline_name, {}).get(test_id, "")

            model_match = classify_match_type(model_output, expected)
            baseline_match = self.baseline_matches.get(test_id, MatchType.NONE)

            raw_score = MATCH_SCORES[model_match]
            difficulty_weight = DIFFICULTY_WEIGHTS[baseline_match]
            weighted_score = raw_score * difficulty_weight
            max_weighted = 1.0 * difficulty_weight

            scores.append(WeightedTestScore(
                test_id=test_id,
                model_name=model_name,
                baseline_name=self.baseline_name,
                category=category,
                model_match=model_match,
                baseline_match=baseline_match,
                raw_score=raw_score,
                difficulty_weight=difficulty_weight,
                weighted_score=weighted_score,
                max_weighted_score=max_weighted,
                expected=expected,
                model_output=model_output,
                baseline_output=baseline_output,
            ))

        return scores

    def get_model_summary(self, model_name: str) -> WeightedModelSummary:
        """Get aggregated summary for a model."""
        scores = self.compute_weighted_scores(model_name)

        n = len(scores)
        if n == 0:
            return WeightedModelSummary(
                model_name=model_name,
                baseline_name=self.baseline_name,
            )

        exact = sum(1 for s in scores if s.model_match == MatchType.EXACT)
        contained = sum(1 for s in scores if s.model_match == MatchType.CONTAINED)
        none_count = sum(1 for s in scores if s.model_match == MatchType.NONE)

        raw_sum = sum(s.raw_score for s in scores)
        weighted_sum = sum(s.weighted_score for s in scores)
        weighted_max = sum(s.max_weighted_score for s in scores)

        # Breakdown by baseline difficulty
        by_difficulty: dict[str, DifficultyBreakdown] = {}
        for difficulty_name, weight in [("easy", 1.0), ("medium", 2.0), ("hard", 3.0)]:
            subset = [s for s in scores if s.difficulty_weight == weight]
            if subset:
                by_difficulty[difficulty_name] = DifficultyBreakdown(
                    difficulty=difficulty_name,
                    weight=weight,
                    count=len(subset),
                    exact_count=sum(1 for s in subset if s.model_match == MatchType.EXACT),
                    contained_count=sum(1 for s in subset if s.model_match == MatchType.CONTAINED),
                    none_count=sum(1 for s in subset if s.model_match == MatchType.NONE),
                    weighted_score=sum(s.weighted_score for s in subset),
                    weighted_max=sum(s.max_weighted_score for s in subset),
                )

        # Breakdown by category
        by_category: dict[str, CategoryBreakdown] = {}
        categories = set(s.category for s in scores if s.category)
        for category in categories:
            subset = [s for s in scores if s.category == category]
            if subset:
                by_category[category] = CategoryBreakdown(
                    category=category,
                    count=len(subset),
                    exact_count=sum(1 for s in subset if s.model_match == MatchType.EXACT),
                    contained_count=sum(1 for s in subset if s.model_match == MatchType.CONTAINED),
                    none_count=sum(1 for s in subset if s.model_match == MatchType.NONE),
                    raw_score=sum(s.raw_score for s in subset),
                    weighted_score=sum(s.weighted_score for s in subset),
                    weighted_max=sum(s.max_weighted_score for s in subset),
                )

        return WeightedModelSummary(
            model_name=model_name,
            baseline_name=self.baseline_name,
            total_tests=n,
            exact_count=exact,
            contained_count=contained,
            none_count=none_count,
            raw_score_sum=raw_sum,
            raw_score_max=float(n),
            hard_accuracy=exact / n,
            soft_accuracy=(exact + contained) / n,
            weighted_score_sum=weighted_sum,
            weighted_score_max=weighted_max,
            weighted_accuracy=weighted_sum / weighted_max if weighted_max > 0 else 0.0,
            by_difficulty=by_difficulty,
            by_category=by_category,
            test_scores=scores,
        )

    def get_all_summaries(self) -> dict[str, WeightedModelSummary]:
        """Get summaries for all non-baseline models."""
        summaries = {}
        for model_name in self.model_outputs.keys():
            if model_name != self.baseline_name:
                summaries[model_name] = self.get_model_summary(model_name)
        return summaries

    def get_baseline_summary(self) -> dict[str, Any]:
        """Get summary of baseline performance (for reference)."""
        if self.baseline_name not in self.model_outputs:
            return {"baseline_name": self.baseline_name, "total_tests": 0}

        n = len(self.baseline_matches)
        exact = sum(1 for m in self.baseline_matches.values() if m == MatchType.EXACT)
        contained = sum(1 for m in self.baseline_matches.values() if m == MatchType.CONTAINED)
        none_count = sum(1 for m in self.baseline_matches.values() if m == MatchType.NONE)

        return {
            "baseline_name": self.baseline_name,
            "total_tests": n,
            "exact_count": exact,
            "contained_count": contained,
            "none_count": none_count,
            "hard_accuracy": exact / n if n > 0 else 0.0,
            "soft_accuracy": (exact + contained) / n if n > 0 else 0.0,
            "difficulty_distribution": {
                "easy": exact,      # Tests where baseline got EXACT
                "medium": contained, # Tests where baseline got CONTAINED
                "hard": none_count,  # Tests where baseline got NONE
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize full scorer state to dictionary."""
        return {
            "baseline_name": self.baseline_name,
            "baseline_summary": self.get_baseline_summary(),
            "model_summaries": {
                name: summary.to_dict()
                for name, summary in self.get_all_summaries().items()
            },
            "test_count": len(self.tests),
            "model_count": len(self.model_outputs),
        }
