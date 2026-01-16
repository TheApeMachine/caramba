"""
Comprehensive scoring framework for behavioral evaluation.

Provides:
1. Binary metrics (exact match, content match, choice correct)
2. Soft scoring (nuanced 5-level scale)
3. Diagnostic flags (repetition loops, distractor contamination)
4. Attention-based metrics (entropy, sparsity)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class SoftScore(IntEnum):
    """Nuanced scoring levels."""
    RADICAL_FAILURE = -1  # Loops, garbage, completely wrong
    WRONG_CONTENT = 0     # Wrong answer or distractor contamination
    CONTENT_BURIED = 1    # Correct content buried in noise
    CONTENT_CORRECT = 2   # Correct with minor format issues
    EXACT_MATCH = 3       # Perfect match


@dataclass
class DiagnosticFlags:
    """Diagnostic flags for failure mode analysis."""
    repetition_loop: bool = False
    distractor_contamination: bool = False
    format_continuation: bool = False
    truncated_output: bool = False
    empty_output: bool = False


@dataclass
class AttentionMetrics:
    """Attention-based diagnostic metrics."""
    entropy: float = 0.0           # Average attention entropy
    sparsity: float = 0.0          # Fraction of near-zero attention
    peak_concentration: float = 0.0  # Mass in top-k positions
    layer_variance: float = 0.0    # Variance across layers


@dataclass
class ModelOutput:
    """Raw output from a model for a single test."""
    model_id: str
    test_id: str
    output_text: str
    logprobs: dict[str, float] | None = None  # For choice tasks
    attention_weights: np.ndarray | None = None  # [layers, heads, seq, seq]
    generation_time_ms: float = 0.0


@dataclass
class TestScore:
    """Complete scoring result for a single test/model pair."""
    model_id: str
    test_id: str

    # Binary metrics
    exact_match: bool = False
    content_match: bool = False
    prefix_match: bool = False
    choice_correct: bool = False  # For multiple choice

    # Soft score
    soft_score: SoftScore = SoftScore.WRONG_CONTENT
    soft_notes: str = ""

    # Diagnostics
    flags: DiagnosticFlags = field(default_factory=DiagnosticFlags)
    attention_metrics: AttentionMetrics | None = None

    # Raw data
    expected: str = ""
    actual: str = ""


@dataclass
class ComparisonResult:
    """Head-to-head comparison between two models."""
    model_a: str
    model_b: str
    test_id: str

    # Who won?
    winner: str | None = None  # model_a, model_b, or None for tie
    score_diff: int = 0        # model_a_score - model_b_score

    # Detailed comparison
    model_a_score: TestScore | None = None
    model_b_score: TestScore | None = None


class Scorer:
    """
    Scores model outputs with binary, soft, and diagnostic metrics.
    """

    def __init__(
        self,
        repetition_threshold: int = 3,
        content_length_ratio: float = 2.0,
    ):
        self.repetition_threshold = repetition_threshold
        self.content_length_ratio = content_length_ratio

    def score(
        self,
        output: ModelOutput,
        expected: str,
        prompt: str,
        choices: list[str] | None = None,
    ) -> TestScore:
        """
        Score a single model output.

        Args:
            output: The model's output
            expected: Expected answer
            prompt: Original prompt (for distractor detection)
            choices: For multiple choice, the available options

        Returns:
            Complete TestScore with all metrics
        """
        result = TestScore(
            model_id=output.model_id,
            test_id=output.test_id,
            expected=expected,
            actual=output.output_text,
        )

        actual = output.output_text.strip()
        expected_str = str(expected).strip()

        # Check for empty output
        if not actual:
            result.flags.empty_output = True
            result.soft_score = SoftScore.RADICAL_FAILURE
            result.soft_notes = "Empty output"
            return result

        # Binary metrics
        result.exact_match = actual == expected_str
        result.content_match = self._content_match(actual, expected_str)
        result.prefix_match = actual.startswith(expected_str)

        # Multiple choice scoring
        if choices and output.logprobs:
            result.choice_correct = self._score_choice(
                output.logprobs, expected_str, choices
            )

        # Soft scoring with diagnostics
        soft_score, notes, flags = self._soft_score(actual, expected_str, prompt)
        result.soft_score = soft_score
        result.soft_notes = notes
        result.flags = flags

        # Attention metrics if available
        if output.attention_weights is not None:
            result.attention_metrics = self._compute_attention_metrics(
                output.attention_weights
            )

        return result

    def _content_match(self, actual: str, expected: str) -> bool:
        """Check if expected content is present in actual output."""
        # Direct containment
        if expected in actual:
            return True

        # Case-insensitive
        if expected.lower() in actual.lower():
            return True

        # Alphanumeric only
        actual_alpha = re.sub(r'[^\w\s]', '', actual)
        expected_alpha = re.sub(r'[^\w\s]', '', expected)
        if expected_alpha and expected_alpha in actual_alpha:
            return True

        return False

    def _score_choice(
        self,
        logprobs: dict[str, float],
        expected: str,
        choices: list[str],
    ) -> bool:
        """Score a multiple choice question by comparing logprobs."""
        # Find logprob for each choice
        choice_probs = {}
        for choice in choices:
            # Handle variations in spacing
            for key in [choice, choice.strip(), f" {choice.strip()}"]:
                if key in logprobs:
                    choice_probs[choice] = logprobs[key]
                    break

        if not choice_probs:
            return False

        # Expected should have highest logprob
        max_choice = max(choice_probs, key=choice_probs.get)
        return expected in max_choice or max_choice in expected

    def _soft_score(
        self,
        actual: str,
        expected: str,
        prompt: str,
    ) -> tuple[SoftScore, str, DiagnosticFlags]:
        """
        Compute soft score with diagnostic analysis.

        Returns:
            Tuple of (score, notes, flags)
        """
        flags = DiagnosticFlags()

        # Check for repetition loop
        if self._detect_repetition_loop(actual):
            flags.repetition_loop = True
            return SoftScore.RADICAL_FAILURE, "Repetition loop detected", flags

        # Exact match
        if actual == expected:
            return SoftScore.EXACT_MATCH, "Exact match", flags

        # Content present checks
        content_present = self._content_match(actual, expected)

        if content_present:
            # Check format continuation
            if self._detect_format_continuation(actual, prompt, expected):
                flags.format_continuation = True
                return SoftScore.CONTENT_CORRECT, "Content correct, format continuation", flags

            # Check length ratio
            len_ratio = len(actual) / max(len(expected), 1)
            if len_ratio < self.content_length_ratio:
                return SoftScore.CONTENT_CORRECT, "Content correct, minor additions", flags
            else:
                return SoftScore.CONTENT_BURIED, "Content present but buried", flags

        # Check for distractor contamination
        if self._detect_distractor_contamination(actual, prompt, expected):
            flags.distractor_contamination = True
            return SoftScore.WRONG_CONTENT, "Distractor contamination", flags

        # Check for garbage
        if len(actual) > 100 and not any(c.isalnum() for c in actual[:50]):
            return SoftScore.RADICAL_FAILURE, "Garbage output", flags

        return SoftScore.WRONG_CONTENT, "Wrong content", flags

    def _detect_repetition_loop(self, text: str, min_repeats: int = 3) -> bool:
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

    def _detect_format_continuation(
        self,
        actual: str,
        prompt: str,
        expected: str,
    ) -> bool:
        """Detect if model continued prompt format but got right content."""
        format_prefixes = [
            'Input:', 'Output:', 'Text:', 'Copy:', 'Echo:',
            'Sequence:', 'Pattern:', 'Data:', 'Row:',
        ]

        has_prefix = any(
            actual.startswith(p) or actual.lower().startswith(p.lower())
            for p in format_prefixes
        )

        return has_prefix and self._content_match(actual, expected)

    def _detect_distractor_contamination(
        self,
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

    def _compute_attention_metrics(
        self,
        attention: np.ndarray,
    ) -> AttentionMetrics:
        """
        Compute attention-based diagnostic metrics.

        Args:
            attention: Shape [layers, heads, seq_len, seq_len]
        """
        # Average over heads and layers for summary stats
        avg_attention = attention.mean(axis=(0, 1))  # [seq, seq]

        # Entropy (higher = more uniform attention)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(avg_attention * np.log(avg_attention + eps), axis=-1).mean()

        # Sparsity (fraction of attention weights < 0.01)
        sparsity = (attention < 0.01).mean()

        # Peak concentration (fraction of mass in top-5 positions)
        sorted_attn = np.sort(avg_attention, axis=-1)[:, ::-1]
        top_k_mass = sorted_attn[:, :5].sum(axis=-1).mean()

        # Layer variance (how much attention patterns vary across layers)
        layer_means = attention.mean(axis=(1, 2, 3))
        layer_variance = layer_means.var()

        return AttentionMetrics(
            entropy=float(entropy),
            sparsity=float(sparsity),
            peak_concentration=float(top_k_mass),
            layer_variance=float(layer_variance),
        )


class MultiModelScorer:
    """
    Scores and compares multiple models on a test suite.
    """

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
        """Add and score a result for one model/test pair."""
        score = self.scorer.score(output, expected, prompt, choices)
        self.scores[output.model_id][output.test_id] = score
        return score

    def compare_pair(
        self,
        model_a: str,
        model_b: str,
        test_id: str,
    ) -> ComparisonResult:
        """Compare two models on a single test."""
        score_a = self.scores[model_a].get(test_id)
        score_b = self.scores[model_b].get(test_id)

        if not score_a or not score_b:
            return ComparisonResult(model_a, model_b, test_id)

        diff = score_a.soft_score - score_b.soft_score

        if diff > 0:
            winner = model_a
        elif diff < 0:
            winner = model_b
        else:
            winner = None

        return ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            test_id=test_id,
            winner=winner,
            score_diff=diff,
            model_a_score=score_a,
            model_b_score=score_b,
        )

    def get_summary(self, model_id: str) -> dict[str, Any]:
        """Get summary statistics for a model."""
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
            "format_continuation": sum(1 for s in scores if s.flags.format_continuation),
            "score_distribution": {
                score.name: sum(1 for s in scores if s.soft_score == score)
                for score in SoftScore
            },
        }

    def get_head_to_head(self, model_a: str, model_b: str) -> dict[str, Any]:
        """Get head-to-head comparison between two models."""
        common_tests = set(self.scores[model_a].keys()) & set(self.scores[model_b].keys())

        wins_a = 0
        wins_b = 0
        ties = 0

        for test_id in common_tests:
            comparison = self.compare_pair(model_a, model_b, test_id)
            if comparison.winner == model_a:
                wins_a += 1
            elif comparison.winner == model_b:
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
        """Get head-to-head comparisons for all model pairs."""
        comparisons = []
        for i, model_a in enumerate(self.model_ids):
            for model_b in self.model_ids[i + 1:]:
                comparisons.append(self.get_head_to_head(model_a, model_b))
        return comparisons
