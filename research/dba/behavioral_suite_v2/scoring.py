"""
Simplified scoring framework for behavioral evaluation.

Tracks match quality per model (none, partial, exact) and computes
teacher vs student comparisons on demand. Designed for undertrained
models where we need to compare relative performance.

Match levels:
- NONE: Expected content not found in output
- PARTIAL: Expected content found with extra prefix/suffix tokens
- EXACT: Output exactly matches expected (after stripping whitespace)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MatchQuality(IntEnum):
    """Match quality levels for model output."""
    NONE = 0     # Expected content not found
    PARTIAL = 1  # Expected found with extra tokens (prefix/suffix)
    EXACT = 2    # Perfect match


@dataclass
class DiagnosticFlags:
    """Diagnostic flags for failure mode analysis."""
    repetition_loop: bool = False
    distractor_contamination: bool = False
    empty_output: bool = False
    format_continuation: bool = False


@dataclass
class AttentionMetrics:
    """Attention-based metrics for a single test result."""
    entropy: float = 0.0  # Higher = more uniform attention
    sparsity: float = 0.0  # Fraction of near-zero weights
    peak_concentration: float = 0.0  # Max attention weight
    diagonal_ratio: float = 0.0  # Self-attention ratio
    head_agreement: float = 0.0  # Similarity between heads
    layer_variance: float = 0.0  # Variance across layers

    # Attack-specific metrics
    target_attention: float = 0.0  # Attention to target token
    distractor_attention: float = 0.0  # Attention to distractor tokens
    attention_shift: float = 0.0  # Change from baseline (if available)


@dataclass
class DegenerationMetrics:
    """Metrics for tracking output degeneration/repetition."""
    unique_token_ratio: float = 1.0  # 1.0 = all unique, 0.0 = all same
    max_consecutive_repeat: int = 0  # Longest repeat sequence
    repetition_ratio: float = 0.0  # Fraction of tokens that are repeats
    total_tokens: int = 0


@dataclass
class MatchResult:
    """Match result for a single model output on a single test."""
    model_id: str
    test_id: str
    quality: MatchQuality
    flags: DiagnosticFlags = field(default_factory=DiagnosticFlags)

    # Raw data for debugging
    expected: str = ""
    actual: str = ""
    first_line: str = ""  # First line of output (often the answer)

    # Extended metrics (optional)
    attention_metrics: AttentionMetrics | None = None
    degeneration_metrics: DegenerationMetrics | None = None


def classify_match(
    output: str,
    expected: str,
    prompt: str = "",
) -> tuple[MatchQuality, DiagnosticFlags, str]:
    """
    Classify match quality between output and expected answer.

    Args:
        output: Raw model output
        expected: Expected answer
        prompt: Original prompt (for distractor detection)

    Returns:
        Tuple of (match_quality, diagnostic_flags, first_line)
    """
    flags = DiagnosticFlags()

    actual = output.strip()
    expected_clean = str(expected).strip()

    # Extract first line for comparison
    first_line = actual.split('\n')[0].strip() if actual else ""

    # Check for empty output
    if not actual:
        flags.empty_output = True
        return MatchQuality.NONE, flags, first_line

    # Check for repetition loop
    if _detect_repetition_loop(actual):
        flags.repetition_loop = True
        return MatchQuality.NONE, flags, first_line

    # Check for distractor contamination
    if prompt and _detect_distractor_contamination(actual, prompt, expected_clean):
        flags.distractor_contamination = True
        return MatchQuality.NONE, flags, first_line

    # EXACT: Full output matches expected exactly
    if actual == expected_clean:
        return MatchQuality.EXACT, flags, first_line

    # EXACT: First line matches expected exactly
    if first_line == expected_clean:
        return MatchQuality.EXACT, flags, first_line

    # PARTIAL: Expected is prefix of output (output has suffix tokens)
    if actual.startswith(expected_clean):
        return MatchQuality.PARTIAL, flags, first_line

    # PARTIAL: Expected is suffix of output (output has prefix tokens)
    if actual.endswith(expected_clean):
        return MatchQuality.PARTIAL, flags, first_line

    # PARTIAL: First line starts with expected
    if first_line.startswith(expected_clean):
        return MatchQuality.PARTIAL, flags, first_line

    # PARTIAL: First line ends with expected
    if first_line.endswith(expected_clean):
        return MatchQuality.PARTIAL, flags, first_line

    # PARTIAL: Expected found somewhere in output (with both prefix and suffix)
    # But be careful with very short expected values - require word boundaries
    if expected_clean in actual:
        # For short expected values (<=3 chars), require it appears as a distinct token
        # not just embedded in random garbage
        if len(expected_clean) <= 3:
            # Check if it appears at start of first line or as isolated word
            words = first_line.split()
            if expected_clean in words or first_line.startswith(expected_clean):
                return MatchQuality.PARTIAL, flags, first_line
            # Also check full output for word boundary
            if re.search(r'(?:^|\s)' + re.escape(expected_clean) + r'(?:\s|$)', actual):
                return MatchQuality.PARTIAL, flags, first_line
            # Short string embedded in garbage - don't count as partial
        else:
            return MatchQuality.PARTIAL, flags, first_line

    # NONE: Expected not found
    return MatchQuality.NONE, flags, first_line


def _detect_repetition_loop(text: str, min_repeats: int = 3) -> bool:
    """Detect pathological repetition in output."""
    if len(text) < 20:
        return False

    # Word-level repetition
    words = text.split()
    if len(words) >= min_repeats * 3:
        for phrase_len in range(1, 5):
            for i in range(len(words) - phrase_len * min_repeats):
                phrase = tuple(words[i:i + phrase_len])
                count = 0
                j = i
                while j <= len(words) - phrase_len:
                    if tuple(words[j:j + phrase_len]) == phrase:
                        count += 1
                        j += phrase_len
                    else:
                        break
                if count >= min_repeats:
                    return True

    # Character-level repetition
    if re.search(r'(.{2,10})\1{3,}', text):
        return True

    return False


def compute_degeneration_metrics(text: str) -> DegenerationMetrics:
    """Compute degeneration/repetition metrics for output text."""
    tokens = text.split()

    if not tokens:
        return DegenerationMetrics(unique_token_ratio=0.0, total_tokens=0)

    n = len(tokens)

    # Unique token ratio
    unique_ratio = len(set(tokens)) / n

    # Find maximum consecutive repetition
    max_repeat = 1
    current_repeat = 1
    for i in range(1, n):
        if tokens[i] == tokens[i - 1]:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1

    # Repetition ratio (fraction of tokens that are repeats)
    repeat_count = sum(1 for i in range(1, n) if tokens[i] == tokens[i - 1])
    repetition_ratio = repeat_count / max(n - 1, 1)

    return DegenerationMetrics(
        unique_token_ratio=unique_ratio,
        max_consecutive_repeat=max_repeat,
        repetition_ratio=repetition_ratio,
        total_tokens=n,
    )


def _detect_distractor_contamination(
    actual: str,
    prompt: str,
    expected: str,
) -> bool:
    """Check if output contains distractor content instead of target."""
    lines = prompt.strip().split('\n')

    distractors = []
    for line in lines[:-2]:
        for pattern in [
            r'Output:\s*(.+?)\.?\s*$',
            r'Copy:\s*(.+?)\.?\s*$',
            r'Echo:\s*(.+?)\.?\s*$',
            r'Repeat:\s*(.+?)\.?\s*$',
            r'->\s*(.+?)\.?\s*$',
            r':\s*(.+?)\.?\s*$',
        ]:
            match = re.search(pattern, line)
            if match:
                distractor = match.group(1).strip()
                if distractor and distractor != expected:
                    distractors.append(distractor)

    actual_lower = actual.lower()
    expected_lower = expected.lower()

    for distractor in distractors:
        if distractor.lower() in actual_lower and expected_lower not in actual_lower:
            return True

    return False


@dataclass
class TestResult:
    """Stores match results for all models on a single test."""
    test_id: str
    expected: str
    prompt: str
    results: dict[str, MatchResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # Test metadata (attack_type, etc.)

    def add_result(
        self,
        model_id: str,
        output: str,
        track_degeneration: bool = True,
        attention_metrics: AttentionMetrics | None = None,
    ) -> MatchResult:
        """
        Add a model's output and classify its match quality.

        Args:
            model_id: Identifier for the model
            output: Raw output text
            track_degeneration: Whether to compute degeneration metrics
            attention_metrics: Pre-computed attention metrics (if available)

        Returns:
            MatchResult with quality classification and metrics
        """
        quality, flags, first_line = classify_match(output, self.expected, self.prompt)

        # Compute degeneration metrics if requested
        degen_metrics = None
        if track_degeneration:
            degen_metrics = compute_degeneration_metrics(output.strip())

        result = MatchResult(
            model_id=model_id,
            test_id=self.test_id,
            quality=quality,
            flags=flags,
            expected=self.expected,
            actual=output.strip(),
            first_line=first_line,
            attention_metrics=attention_metrics,
            degeneration_metrics=degen_metrics,
        )
        self.results[model_id] = result
        return result

    def get_quality(self, model_id: str) -> MatchQuality:
        """Get match quality for a model, defaulting to NONE if not found."""
        if model_id not in self.results:
            return MatchQuality.NONE
        return self.results[model_id].quality

    def is_adversarial(self) -> bool:
        """Check if this test is an adversarial attack test."""
        return self.metadata.get("attack_type") is not None


class ComparisonOutcome(IntEnum):
    """Outcome of comparing student vs teacher on a single test."""
    STUDENT_WORSE = -1   # Student has lower quality than teacher
    TIE = 0              # Same quality
    STUDENT_BETTER = 1   # Student has higher quality than teacher


@dataclass
class PairComparison:
    """Comparison result between teacher and student on a single test."""
    test_id: str
    teacher_quality: MatchQuality
    student_quality: MatchQuality
    outcome: ComparisonOutcome

    # Flags for interesting cases
    student_exact_teacher_partial: bool = False  # Student exact, teacher has extra tokens
    student_exact_teacher_none: bool = False     # Student exact, teacher failed
    both_exact: bool = False                     # Both got it exactly right
    both_partial: bool = False                   # Both got it with extra tokens
    both_none: bool = False                      # Both failed


def compare_teacher_student(
    test_result: TestResult,
    teacher_id: str,
    student_id: str,
) -> PairComparison:
    """
    Compare teacher and student on a single test.

    Returns comparison with outcome and flags for interesting cases.
    """
    t_quality = test_result.get_quality(teacher_id)
    s_quality = test_result.get_quality(student_id)

    # Determine outcome
    if s_quality > t_quality:
        outcome = ComparisonOutcome.STUDENT_BETTER
    elif s_quality < t_quality:
        outcome = ComparisonOutcome.STUDENT_WORSE
    else:
        outcome = ComparisonOutcome.TIE

    comparison = PairComparison(
        test_id=test_result.test_id,
        teacher_quality=t_quality,
        student_quality=s_quality,
        outcome=outcome,
    )

    # Set flags for interesting cases
    comparison.student_exact_teacher_partial = (
        s_quality == MatchQuality.EXACT and t_quality == MatchQuality.PARTIAL
    )
    comparison.student_exact_teacher_none = (
        s_quality == MatchQuality.EXACT and t_quality == MatchQuality.NONE
    )
    comparison.both_exact = (
        s_quality == MatchQuality.EXACT and t_quality == MatchQuality.EXACT
    )
    comparison.both_partial = (
        s_quality == MatchQuality.PARTIAL and t_quality == MatchQuality.PARTIAL
    )
    comparison.both_none = (
        s_quality == MatchQuality.NONE and t_quality == MatchQuality.NONE
    )

    return comparison


class BehavioralScorer:
    """
    Manages scoring across multiple tests and models.

    Stores raw match results and computes summaries/comparisons on demand.
    """

    def __init__(self):
        self.tests: dict[str, TestResult] = {}
        self.model_ids: set[str] = set()

    def add_test(
        self,
        test_id: str,
        expected: str,
        prompt: str = "",
        metadata: dict | None = None,
    ) -> TestResult:
        """Register a new test case with optional metadata."""
        test_result = TestResult(
            test_id=test_id,
            expected=expected,
            prompt=prompt,
            metadata=metadata or {},
        )
        self.tests[test_id] = test_result
        return test_result

    def add_output(
        self,
        test_id: str,
        model_id: str,
        output: str,
        track_degeneration: bool = True,
        attention_metrics: AttentionMetrics | None = None,
    ) -> MatchResult:
        """Add a model's output for a test."""
        if test_id not in self.tests:
            raise KeyError(f"Test {test_id} not registered. Call add_test first.")
        self.model_ids.add(model_id)
        return self.tests[test_id].add_result(
            model_id,
            output,
            track_degeneration=track_degeneration,
            attention_metrics=attention_metrics,
        )

    def get_model_summary(self, model_id: str) -> dict[str, Any]:
        """Get summary statistics for a single model."""
        results = [
            test.results[model_id]
            for test in self.tests.values()
            if model_id in test.results
        ]

        if not results:
            return {"model_id": model_id, "total_tests": 0}

        n = len(results)
        exact_count = sum(1 for r in results if r.quality == MatchQuality.EXACT)
        partial_count = sum(1 for r in results if r.quality == MatchQuality.PARTIAL)
        none_count = sum(1 for r in results if r.quality == MatchQuality.NONE)

        summary = {
            "model_id": model_id,
            "total_tests": n,
            "exact_match_count": exact_count,
            "partial_match_count": partial_count,
            "no_match_count": none_count,
            "exact_match_rate": exact_count / n,
            "partial_or_better_rate": (exact_count + partial_count) / n,
            "quality_distribution": {
                "EXACT": exact_count,
                "PARTIAL": partial_count,
                "NONE": none_count,
            },
            # Diagnostic counts
            "repetition_loops": sum(1 for r in results if r.flags.repetition_loop),
            "distractor_contamination": sum(1 for r in results if r.flags.distractor_contamination),
            "empty_outputs": sum(1 for r in results if r.flags.empty_output),
        }

        # Add degeneration statistics if available
        degen_results = [r for r in results if r.degeneration_metrics is not None]
        if degen_results:
            avg_unique_ratio = sum(
                r.degeneration_metrics.unique_token_ratio for r in degen_results
            ) / len(degen_results)
            avg_repeat_ratio = sum(
                r.degeneration_metrics.repetition_ratio for r in degen_results
            ) / len(degen_results)
            max_consecutive = max(
                r.degeneration_metrics.max_consecutive_repeat for r in degen_results
            )
            high_degen_count = sum(
                1 for r in degen_results
                if r.degeneration_metrics.unique_token_ratio < 0.5
            )

            summary["degeneration_stats"] = {
                "avg_unique_token_ratio": avg_unique_ratio,
                "avg_repetition_ratio": avg_repeat_ratio,
                "max_consecutive_repeat": max_consecutive,
                "high_degeneration_count": high_degen_count,
            }

        # Add attention statistics if available
        attn_results = [r for r in results if r.attention_metrics is not None]
        if attn_results:
            summary["attention_stats"] = {
                "avg_entropy": sum(r.attention_metrics.entropy for r in attn_results) / len(attn_results),
                "avg_sparsity": sum(r.attention_metrics.sparsity for r in attn_results) / len(attn_results),
                "avg_peak_concentration": sum(r.attention_metrics.peak_concentration for r in attn_results) / len(attn_results),
            }

        return summary

    def compare_models(
        self,
        teacher_id: str,
        student_id: str,
    ) -> dict[str, Any]:
        """
        Get head-to-head comparison between teacher and student.

        Returns summary with win/loss/tie counts and breakdowns by case type.
        """
        comparisons = []
        for test in self.tests.values():
            if teacher_id in test.results and student_id in test.results:
                comparisons.append(compare_teacher_student(test, teacher_id, student_id))

        if not comparisons:
            return {
                "teacher_id": teacher_id,
                "student_id": student_id,
                "total_tests": 0,
            }

        n = len(comparisons)
        student_wins = sum(1 for c in comparisons if c.outcome == ComparisonOutcome.STUDENT_BETTER)
        teacher_wins = sum(1 for c in comparisons if c.outcome == ComparisonOutcome.STUDENT_WORSE)
        ties = sum(1 for c in comparisons if c.outcome == ComparisonOutcome.TIE)

        return {
            "teacher_id": teacher_id,
            "student_id": student_id,
            "total_tests": n,
            "student_wins": student_wins,
            "teacher_wins": teacher_wins,
            "ties": ties,
            "student_win_rate": student_wins / n,
            "teacher_win_rate": teacher_wins / n,
            # Interesting case breakdowns
            "both_exact": sum(1 for c in comparisons if c.both_exact),
            "both_partial": sum(1 for c in comparisons if c.both_partial),
            "both_none": sum(1 for c in comparisons if c.both_none),
            "student_exact_teacher_partial": sum(1 for c in comparisons if c.student_exact_teacher_partial),
            "student_exact_teacher_none": sum(1 for c in comparisons if c.student_exact_teacher_none),
        }

    def get_detailed_comparisons(
        self,
        teacher_id: str,
        student_id: str,
    ) -> list[PairComparison]:
        """Get detailed per-test comparisons between teacher and student."""
        comparisons = []
        for test in self.tests.values():
            if teacher_id in test.results and student_id in test.results:
                comparisons.append(compare_teacher_student(test, teacher_id, student_id))
        return comparisons

    def get_interesting_cases(
        self,
        teacher_id: str,
        student_id: str,
    ) -> dict[str, list[str]]:
        """
        Get test IDs grouped by interesting comparison outcomes.

        Useful for finding cases where student outperforms teacher or vice versa.
        """
        comparisons = self.get_detailed_comparisons(teacher_id, student_id)

        return {
            "student_exact_teacher_partial": [
                c.test_id for c in comparisons if c.student_exact_teacher_partial
            ],
            "student_exact_teacher_none": [
                c.test_id for c in comparisons if c.student_exact_teacher_none
            ],
            "teacher_exact_student_partial": [
                c.test_id for c in comparisons
                if c.teacher_quality == MatchQuality.EXACT and c.student_quality == MatchQuality.PARTIAL
            ],
            "teacher_exact_student_none": [
                c.test_id for c in comparisons
                if c.teacher_quality == MatchQuality.EXACT and c.student_quality == MatchQuality.NONE
            ],
            "both_exact": [c.test_id for c in comparisons if c.both_exact],
            "both_none": [c.test_id for c in comparisons if c.both_none],
        }

    def get_adversarial_summary(self, model_id: str) -> dict[str, Any]:
        """
        Get summary statistics specifically for adversarial tests.

        Groups results by attack type and reports robustness metrics.
        """
        # Filter to adversarial tests with this model
        adv_tests = [
            (test, test.results[model_id])
            for test in self.tests.values()
            if model_id in test.results and test.is_adversarial()
        ]

        if not adv_tests:
            return {"model_id": model_id, "total_adversarial_tests": 0}

        # Group by attack type
        by_attack_type: dict[str, list] = {}
        for test, result in adv_tests:
            attack_type = test.metadata.get("attack_type", "unknown")
            if attack_type not in by_attack_type:
                by_attack_type[attack_type] = []
            by_attack_type[attack_type].append(result)

        # Compute per-attack-type stats
        attack_stats = {}
        for attack_type, results in by_attack_type.items():
            n = len(results)
            exact = sum(1 for r in results if r.quality == MatchQuality.EXACT)
            partial = sum(1 for r in results if r.quality == MatchQuality.PARTIAL)

            attack_stats[attack_type] = {
                "total": n,
                "exact_count": exact,
                "partial_count": partial,
                "robustness_rate": exact / n,  # Exact match = resisted attack
                "partial_or_better": (exact + partial) / n,
            }

        # Overall adversarial robustness
        all_results = [r for _, r in adv_tests]
        n_total = len(all_results)
        n_exact = sum(1 for r in all_results if r.quality == MatchQuality.EXACT)

        return {
            "model_id": model_id,
            "total_adversarial_tests": n_total,
            "overall_robustness_rate": n_exact / n_total,
            "attack_types": attack_stats,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize all results to a dictionary."""

        def serialize_result(result: MatchResult) -> dict:
            data = {
                "quality": result.quality.name,
                "actual": result.actual,
                "first_line": result.first_line,
                "flags": {
                    "repetition_loop": result.flags.repetition_loop,
                    "distractor_contamination": result.flags.distractor_contamination,
                    "empty_output": result.flags.empty_output,
                },
            }

            # Include degeneration metrics if present
            if result.degeneration_metrics:
                data["degeneration"] = {
                    "unique_token_ratio": result.degeneration_metrics.unique_token_ratio,
                    "max_consecutive_repeat": result.degeneration_metrics.max_consecutive_repeat,
                    "repetition_ratio": result.degeneration_metrics.repetition_ratio,
                    "total_tokens": result.degeneration_metrics.total_tokens,
                }

            # Include attention metrics if present
            if result.attention_metrics:
                data["attention"] = {
                    "entropy": result.attention_metrics.entropy,
                    "sparsity": result.attention_metrics.sparsity,
                    "peak_concentration": result.attention_metrics.peak_concentration,
                    "diagonal_ratio": result.attention_metrics.diagonal_ratio,
                }

            return data

        return {
            "tests": {
                test_id: {
                    "expected": test.expected,
                    "metadata": test.metadata,
                    "results": {
                        model_id: serialize_result(result)
                        for model_id, result in test.results.items()
                    },
                }
                for test_id, test in self.tests.items()
            },
            "model_ids": list(self.model_ids),
        }


# =============================================================================
# Legacy compatibility - keep old classes available but deprecated
# =============================================================================

# These are kept for backwards compatibility with existing code that imports them

class SoftScore(IntEnum):
    """DEPRECATED: Use MatchQuality instead."""
    RADICAL_FAILURE = -1
    WRONG_CONTENT = 0
    CONTENT_BURIED = 1
    CONTENT_CORRECT = 2
    EXACT_MATCH = 3


@dataclass
class ModelOutput:
    """Raw output from a model for a single test."""
    model_id: str
    test_id: str
    output_text: str
    logprobs: dict[str, float] | None = None
    attention_weights: Any = None
    generation_time_ms: float = 0.0


@dataclass
class TestScore:
    """DEPRECATED: Use MatchResult instead."""
    model_id: str
    test_id: str
    exact_match: bool = False
    content_match: bool = False
    prefix_match: bool = False
    choice_correct: bool = False
    soft_score: SoftScore = SoftScore.WRONG_CONTENT
    soft_notes: str = ""
    flags: DiagnosticFlags = field(default_factory=DiagnosticFlags)
    attention_metrics: Any = None
    expected: str = ""
    actual: str = ""


class Scorer:
    """DEPRECATED: Use BehavioralScorer instead."""

    def __init__(self, **kwargs):
        pass

    def score(
        self,
        output: ModelOutput,
        expected: str,
        prompt: str,
        choices: list[str] | None = None,
    ) -> TestScore:
        """Score using new system but return old-style TestScore."""
        quality, flags, first_line = classify_match(output.output_text, expected, prompt)

        # Map new quality to old soft score
        quality_to_soft = {
            MatchQuality.NONE: SoftScore.WRONG_CONTENT,
            MatchQuality.PARTIAL: SoftScore.CONTENT_CORRECT,
            MatchQuality.EXACT: SoftScore.EXACT_MATCH,
        }

        # Check for repetition loop -> RADICAL_FAILURE
        if flags.repetition_loop:
            soft = SoftScore.RADICAL_FAILURE
        else:
            soft = quality_to_soft.get(quality, SoftScore.WRONG_CONTENT)

        return TestScore(
            model_id=output.model_id,
            test_id=output.test_id,
            exact_match=(quality == MatchQuality.EXACT),
            content_match=(quality >= MatchQuality.PARTIAL),
            prefix_match=output.output_text.strip().startswith(expected.strip()),
            soft_score=soft,
            soft_notes=f"Quality: {quality.name}",
            flags=flags,
            expected=expected,
            actual=output.output_text,
        )


class MultiModelScorer:
    """DEPRECATED: Use BehavioralScorer instead."""

    def __init__(self, model_ids: list[str]):
        self.model_ids = model_ids
        self.scorer = Scorer()
        self.scores: dict[str, dict[str, TestScore]] = {
            mid: {} for mid in model_ids
        }

    def add_result(
        self,
        output: ModelOutput,
        expected: str,
        prompt: str,
        choices: list[str] | None = None,
    ) -> TestScore:
        score = self.scorer.score(output, expected, prompt, choices)
        self.scores[output.model_id][output.test_id] = score
        return score

    def get_summary(self, model_id: str) -> dict[str, Any]:
        scores = list(self.scores[model_id].values())
        if not scores:
            return {}

        n = len(scores)
        return {
            "model_id": model_id,
            "total_tests": n,
            "exact_match_rate": sum(1 for s in scores if s.exact_match) / n,
            "content_match_rate": sum(1 for s in scores if s.content_match) / n,
            "soft_score_avg": sum(s.soft_score for s in scores) / n,
            "soft_score_total": sum(s.soft_score for s in scores),
            "repetition_loops": sum(1 for s in scores if s.flags.repetition_loop),
            "distractor_contamination": sum(1 for s in scores if s.flags.distractor_contamination),
            "format_continuation": 0,
            "score_distribution": {
                score.name: sum(1 for s in scores if s.soft_score == score)
                for score in SoftScore
            },
        }

    def get_head_to_head(self, model_a: str, model_b: str) -> dict[str, Any]:
        common_tests = set(self.scores[model_a].keys()) & set(self.scores[model_b].keys())

        wins_a = 0
        wins_b = 0
        ties = 0

        for test_id in common_tests:
            score_a = self.scores[model_a][test_id]
            score_b = self.scores[model_b][test_id]
            diff = score_a.soft_score - score_b.soft_score
            if diff > 0:
                wins_a += 1
            elif diff < 0:
                wins_b += 1
            else:
                ties += 1

        return {
            "model_a": model_a,
            "model_b": model_b,
            "total_tests": len(common_tests),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / len(common_tests) if common_tests else 0,
            "win_rate_b": wins_b / len(common_tests) if common_tests else 0,
        }

    def get_all_comparisons(self) -> list[dict[str, Any]]:
        comparisons = []
        for i, model_a in enumerate(self.model_ids):
            for model_b in self.model_ids[i + 1:]:
                comparisons.append(self.get_head_to_head(model_a, model_b))
        return comparisons
