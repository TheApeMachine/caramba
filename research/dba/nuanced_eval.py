"""
Nuanced behavioral evaluation for DBA paper.

Extends the standard eval framework with a multi-level scoring system
that distinguishes between exact matches, content-correct responses,
and various failure modes (distractor contamination, repetition loops).

Scoring system:
  3 = Exact match
  2 = Content correct, minor format additions (prefix/suffix)
  1 = Content present but buried in noise
  0 = Wrong content OR distractor contamination
 -1 = Radical failure (garbage, infinite loop, completely unrelated)

Additional diagnostics:
  - repetition_loop: Did the model enter a pathological loop?
  - distractor_contamination: Did the model output earlier examples instead of target?
  - format_continuation: Did model continue prompt structure but get right content?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class NuancedScore(IntEnum):
    """Nuanced scoring levels for behavioral evaluation."""
    RADICAL_FAILURE = -1  # Garbage, infinite loop, completely unrelated
    WRONG_CONTENT = 0     # Wrong content or distractor contamination
    CONTENT_BURIED = 1    # Content present but buried in noise
    CONTENT_CORRECT = 2   # Content correct, minor format additions
    EXACT_MATCH = 3       # Exact match


@dataclass
class NuancedFlags:
    """Diagnostic flags for a single model output."""
    repetition_loop: bool = False
    distractor_contamination: bool = False
    format_continuation: bool = False


@dataclass
class NuancedCaseResult:
    """Records nuanced per-case outcomes for teacher and student."""
    case_id: str
    prompt: str
    expected: str

    teacher_output: str
    student_output: str

    teacher_strict_pass: bool
    student_strict_pass: bool

    teacher_score: NuancedScore = NuancedScore.WRONG_CONTENT
    student_score: NuancedScore = NuancedScore.WRONG_CONTENT

    teacher_flags: NuancedFlags = field(default_factory=NuancedFlags)
    student_flags: NuancedFlags = field(default_factory=NuancedFlags)

    teacher_notes: str = ""
    student_notes: str = ""


@dataclass
class NuancedSummary:
    """Aggregates nuanced evaluation metrics across all test cases."""
    results: list[NuancedCaseResult]

    # Strict pass rates (exact match only)
    teacher_strict_accuracy: float = 0.0
    student_strict_accuracy: float = 0.0

    # Content correct rates (score >= 2)
    teacher_content_accuracy: float = 0.0
    student_content_accuracy: float = 0.0

    # Content present rates (score >= 1)
    teacher_present_accuracy: float = 0.0
    student_present_accuracy: float = 0.0

    # Total and average scores
    teacher_total_score: int = 0
    student_total_score: int = 0
    teacher_avg_score: float = 0.0
    student_avg_score: float = 0.0

    # Failure mode counts
    teacher_repetition_loops: int = 0
    student_repetition_loops: int = 0
    teacher_distractor_contaminations: int = 0
    student_distractor_contaminations: int = 0
    teacher_format_continuations: int = 0
    student_format_continuations: int = 0

    # Head-to-head
    student_wins: int = 0
    teacher_wins: int = 0
    ties: int = 0

    def __post_init__(self) -> None:
        """Compute aggregate metrics from individual results."""
        if not self.results:
            return

        n = len(self.results)

        # Strict accuracy
        self.teacher_strict_accuracy = sum(
            1 for r in self.results if r.teacher_strict_pass
        ) / n
        self.student_strict_accuracy = sum(
            1 for r in self.results if r.student_strict_pass
        ) / n

        # Content correct (score >= 2)
        self.teacher_content_accuracy = sum(
            1 for r in self.results if r.teacher_score >= NuancedScore.CONTENT_CORRECT
        ) / n
        self.student_content_accuracy = sum(
            1 for r in self.results if r.student_score >= NuancedScore.CONTENT_CORRECT
        ) / n

        # Content present (score >= 1)
        self.teacher_present_accuracy = sum(
            1 for r in self.results if r.teacher_score >= NuancedScore.CONTENT_BURIED
        ) / n
        self.student_present_accuracy = sum(
            1 for r in self.results if r.student_score >= NuancedScore.CONTENT_BURIED
        ) / n

        # Total and average scores
        self.teacher_total_score = sum(r.teacher_score for r in self.results)
        self.student_total_score = sum(r.student_score for r in self.results)
        self.teacher_avg_score = self.teacher_total_score / n
        self.student_avg_score = self.student_total_score / n

        # Failure modes
        self.teacher_repetition_loops = sum(
            1 for r in self.results if r.teacher_flags.repetition_loop
        )
        self.student_repetition_loops = sum(
            1 for r in self.results if r.student_flags.repetition_loop
        )
        self.teacher_distractor_contaminations = sum(
            1 for r in self.results if r.teacher_flags.distractor_contamination
        )
        self.student_distractor_contaminations = sum(
            1 for r in self.results if r.student_flags.distractor_contamination
        )
        self.teacher_format_continuations = sum(
            1 for r in self.results if r.teacher_flags.format_continuation
        )
        self.student_format_continuations = sum(
            1 for r in self.results if r.student_flags.format_continuation
        )

        # Head-to-head
        for r in self.results:
            if r.student_score > r.teacher_score:
                self.student_wins += 1
            elif r.teacher_score > r.student_score:
                self.teacher_wins += 1
            else:
                self.ties += 1


def detect_repetition_loop(output: str, min_repeats: int = 3) -> bool:
    """
    Detect if output contains pathological repetition.

    Checks for:
    - Repeated word sequences (3+ words repeated 3+ times consecutively)
    - Repeated character patterns (regex for 2-10 char sequences repeated 4+ times)

    Args:
        output: The model output to check
        min_repeats: Minimum number of consecutive repeats to flag as a loop

    Returns:
        True if a repetition loop is detected
    """
    if len(output) < 20:
        return False

    # Check for repeated phrases (word sequences repeated consecutively)
    words = output.split()
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

    # Check for repeated character patterns
    if re.search(r'(.{2,10})\1{3,}', output):
        return True

    return False


def detect_distractor_contamination(
    output: str,
    prompt: str,
    expected: str,
) -> bool:
    """
    Check if output contains content from earlier examples instead of target.

    Extracts potential distractors from the prompt (values from earlier
    few-shot examples) and checks if the output contains these distractors
    but not the expected answer.

    Args:
        output: The model output to check
        prompt: The full prompt including few-shot examples
        expected: The expected answer

    Returns:
        True if distractor contamination is detected
    """
    lines = prompt.strip().split('\n')

    # Extract potential distractors from earlier examples
    distractors: list[str] = []
    for line in lines[:-2]:  # Exclude the last line (the query)
        # Common patterns for few-shot example answers
        for pattern in [
            r'Output:\s*(.+?)\.?\s*$',
            r'Copy:\s*(.+?)\.?\s*$',
            r'->\s*(.+?)\.?\s*$',
            r':\s*(.+?)\.?\s*$',
        ]:
            match = re.search(pattern, line)
            if match:
                distractor = match.group(1).strip()
                if distractor and distractor != expected.strip():
                    distractors.append(distractor)

    # Check if output contains distractors but not expected
    output_clean = output.strip().lower()
    expected_clean = expected.strip().lower()

    for distractor in distractors:
        distractor_clean = distractor.lower()
        if distractor_clean in output_clean and expected_clean not in output_clean:
            return True

    return False


def extract_content_match(
    output: str,
    expected: str,
) -> tuple[bool, bool]:
    """
    Check if expected content is present in output.

    Args:
        output: The model output
        expected: The expected answer

    Returns:
        Tuple of (content_present, is_exact_match)
    """
    output_clean = output.strip()
    expected_clean = expected.strip()

    # Exact match
    if output_clean == expected_clean:
        return True, True

    # Content present somewhere in output
    if expected_clean in output_clean:
        return True, False

    # Try case-insensitive
    if expected_clean.lower() in output_clean.lower():
        return True, False

    # Try without punctuation
    output_alphanum = re.sub(r'[^\w\s]', '', output_clean)
    expected_alphanum = re.sub(r'[^\w\s]', '', expected_clean)
    if expected_alphanum and expected_alphanum in output_alphanum:
        return True, False

    return False, False


def detect_format_continuation(
    output: str,
    prompt: str,
    expected: str,
) -> bool:
    """
    Check if model continued prompt format but got right content.

    This detects cases where the model outputs something like:
    "Input: A7 B4 C9 D2. Output:" instead of just "A7 B4 C9 D2."

    Args:
        output: The model output
        prompt: The full prompt
        expected: The expected answer

    Returns:
        True if format continuation with correct content is detected
    """
    # Common format prefixes from few-shot prompts
    format_prefixes = [
        'Input:', 'Output:', 'Text:', 'Copy:', 'X ->', 'X:',
        'Sequence:', 'Pattern:', 'Data:', 'Row:', 'book ->',
        'Start with', 'Take the first',
    ]

    output_clean = output.strip()
    expected_clean = expected.strip()

    # Check if output has format prefix but contains expected content
    has_prefix = any(
        output_clean.startswith(p) or output_clean.lower().startswith(p.lower())
        for p in format_prefixes
    )
    content_present, _ = extract_content_match(output, expected)

    return has_prefix and content_present


def score_output(
    output: str,
    expected: str,
    prompt: str,
) -> tuple[NuancedScore, str, NuancedFlags]:
    """
    Score an output on the nuanced scale.

    Args:
        output: The model output
        expected: The expected answer
        prompt: The full prompt (for distractor detection)

    Returns:
        Tuple of (score, explanation_notes, diagnostic_flags)
    """
    flags = NuancedFlags()
    output_clean = output.strip()
    expected_clean = expected.strip()

    # Check for repetition loop first (radical failure)
    if detect_repetition_loop(output):
        flags.repetition_loop = True
        return NuancedScore.RADICAL_FAILURE, "Repetition loop detected", flags

    # Check for exact match
    if output_clean == expected_clean:
        return NuancedScore.EXACT_MATCH, "Exact match", flags

    # Check if content is present
    content_present, _ = extract_content_match(output, expected)

    if content_present:
        # Check if it's format continuation
        if detect_format_continuation(output, prompt, expected):
            flags.format_continuation = True
            return NuancedScore.CONTENT_CORRECT, "Content correct, format continuation", flags

        # Content present but maybe with extra stuff
        len_ratio = len(output_clean) / max(len(expected_clean), 1)
        if len_ratio < 2.0:
            return NuancedScore.CONTENT_CORRECT, "Content correct, minor additions", flags
        else:
            return NuancedScore.CONTENT_BURIED, "Content present but buried", flags

    # Check for distractor contamination
    if detect_distractor_contamination(output, prompt, expected):
        flags.distractor_contamination = True
        return NuancedScore.WRONG_CONTENT, "Distractor contamination", flags

    # Check for garbage output
    if len(output_clean) > 100 and not any(c.isalnum() for c in output_clean[:50]):
        return NuancedScore.RADICAL_FAILURE, "Garbage output", flags

    # Wrong content
    return NuancedScore.WRONG_CONTENT, "Wrong content", flags


def run_nuanced_eval(
    results: list[tuple[str, str, str, str, str, bool, bool]],
) -> NuancedSummary:
    """
    Run nuanced evaluation on a list of results.

    Args:
        results: List of tuples containing:
            (case_id, prompt, expected, teacher_output, student_output,
             teacher_strict_pass, student_strict_pass)

    Returns:
        NuancedSummary with aggregate metrics
    """
    nuanced_results: list[NuancedCaseResult] = []

    for (case_id, prompt, expected, teacher_out, student_out,
         teacher_strict, student_strict) in results:

        # Score teacher
        t_score, t_notes, t_flags = score_output(teacher_out, expected, prompt)

        # Score student
        s_score, s_notes, s_flags = score_output(student_out, expected, prompt)

        nuanced_results.append(NuancedCaseResult(
            case_id=case_id,
            prompt=prompt,
            expected=expected,
            teacher_output=teacher_out,
            student_output=student_out,
            teacher_strict_pass=teacher_strict,
            student_strict_pass=student_strict,
            teacher_score=t_score,
            student_score=s_score,
            teacher_flags=t_flags,
            student_flags=s_flags,
            teacher_notes=t_notes,
            student_notes=s_notes,
        ))

    return NuancedSummary(results=nuanced_results)


def parse_behavior_log(log_path: str) -> list[tuple[str, str, str, str, str, bool, bool]]:
    """
    Parse a behavior log file into structured results.

    Args:
        log_path: Path to the behavior_log.txt file

    Returns:
        List of tuples: (case_id, prompt, expected, teacher_output,
                        student_output, teacher_strict_pass, student_strict_pass)
    """
    with open(log_path, 'r') as f:
        content = f.read()

    results: list[tuple[str, str, str, str, str, bool, bool]] = []

    # Split by test delimiter
    test_blocks = re.split(r'\n\[(\d+)/\d+\]\s+(\w+)\n-+\n', content)

    # Skip header
    i = 1
    while i < len(test_blocks) - 2:
        test_num = test_blocks[i]
        test_name = test_blocks[i + 1]
        test_content = test_blocks[i + 2]

        # Parse test content
        prompt_match = re.search(
            r'PROMPT:\n(.*?)\n\nEXPECTED:', test_content, re.DOTALL
        )
        expected_match = re.search(
            r'EXPECTED:\s*(.*?)\n\nTEACHER', test_content, re.DOTALL
        )
        teacher_match = re.search(
            r'TEACHER ([✓✗]):\n(.*?)\n\nSTUDENT', test_content, re.DOTALL
        )
        student_match = re.search(
            r'STUDENT ([✓✗]):\n(.*?)(?:\n\n|\n*$)', test_content, re.DOTALL
        )

        if all([prompt_match, expected_match, teacher_match, student_match]):
            results.append((
                test_name,
                prompt_match.group(1).strip(),
                expected_match.group(1).strip(),
                teacher_match.group(2).strip(),
                student_match.group(2).strip(),
                teacher_match.group(1) == '✓',
                student_match.group(1) == '✓',
            ))

        i += 3

    return results


def format_summary_report(summary: NuancedSummary) -> str:
    """
    Format a NuancedSummary as a human-readable report.

    Args:
        summary: The NuancedSummary to format

    Returns:
        Formatted string report
    """
    lines: list[str] = []
    n = len(summary.results)

    lines.append("=" * 100)
    lines.append("NUANCED BEHAVIORAL ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("SCORING LEGEND:")
    lines.append("  3 = Exact match")
    lines.append("  2 = Content correct, minor format additions")
    lines.append("  1 = Content present but buried")
    lines.append("  0 = Wrong content / distractor contamination")
    lines.append(" -1 = Radical failure (loop, garbage)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("OVERALL SCORES")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<40} {'Teacher (Baseline)':<25} {'Student (DBA)':<25}")
    lines.append("-" * 100)

    lines.append(
        f"{'Strict Pass (exact match)':<40} "
        f"{sum(1 for r in summary.results if r.teacher_strict_pass)}/{n} "
        f"({100*summary.teacher_strict_accuracy:.1f}%){'':<5} "
        f"{sum(1 for r in summary.results if r.student_strict_pass)}/{n} "
        f"({100*summary.student_strict_accuracy:.1f}%)"
    )
    lines.append(
        f"{'Content Correct (score >= 2)':<40} "
        f"{sum(1 for r in summary.results if r.teacher_score >= 2)}/{n} "
        f"({100*summary.teacher_content_accuracy:.1f}%){'':<5} "
        f"{sum(1 for r in summary.results if r.student_score >= 2)}/{n} "
        f"({100*summary.student_content_accuracy:.1f}%)"
    )
    lines.append(
        f"{'Content Present (score >= 1)':<40} "
        f"{sum(1 for r in summary.results if r.teacher_score >= 1)}/{n} "
        f"({100*summary.teacher_present_accuracy:.1f}%){'':<5} "
        f"{sum(1 for r in summary.results if r.student_score >= 1)}/{n} "
        f"({100*summary.student_present_accuracy:.1f}%)"
    )
    lines.append(
        f"{'Total Score':<40} "
        f"{summary.teacher_total_score}{'':<20} "
        f"{summary.student_total_score}"
    )
    lines.append(
        f"{'Average Score':<40} "
        f"{summary.teacher_avg_score:.2f}{'':<20} "
        f"{summary.student_avg_score:.2f}"
    )

    lines.append("")
    lines.append("-" * 100)
    lines.append("FAILURE MODE ANALYSIS")
    lines.append("-" * 100)
    lines.append(
        f"{'Repetition Loops':<40} "
        f"{summary.teacher_repetition_loops}{'':<20} "
        f"{summary.student_repetition_loops}"
    )
    lines.append(
        f"{'Distractor Contamination':<40} "
        f"{summary.teacher_distractor_contaminations}{'':<20} "
        f"{summary.student_distractor_contaminations}"
    )
    lines.append(
        f"{'Format Continuation (content correct)':<40} "
        f"{summary.teacher_format_continuations}{'':<20} "
        f"{summary.student_format_continuations}"
    )

    lines.append("")
    lines.append("-" * 100)
    lines.append("HEAD-TO-HEAD COMPARISON")
    lines.append("-" * 100)
    lines.append(f"Student (DBA) wins: {summary.student_wins}")
    lines.append(f"Teacher (Baseline) wins: {summary.teacher_wins}")
    lines.append(f"Ties: {summary.ties}")

    # Detailed wins
    student_wins = [
        (r.case_id, r.student_score - r.teacher_score, r)
        for r in summary.results if r.student_score > r.teacher_score
    ]
    teacher_wins = [
        (r.case_id, r.teacher_score - r.student_score, r)
        for r in summary.results if r.teacher_score > r.student_score
    ]

    if student_wins:
        lines.append("")
        lines.append("-" * 100)
        lines.append("DETAILED: STUDENT (DBA) WINS")
        lines.append("-" * 100)
        for name, margin, r in sorted(student_wins, key=lambda x: -x[1]):
            lines.append(f"\n[{name}] (margin: +{margin})")
            lines.append(f"  Expected: {r.expected[:60]}{'...' if len(r.expected) > 60 else ''}")
            lines.append(f"  Teacher ({r.teacher_score}): {r.teacher_output[:60]}{'...' if len(r.teacher_output) > 60 else ''}")
            lines.append(f"  Student ({r.student_score}): {r.student_output[:60]}{'...' if len(r.student_output) > 60 else ''}")
            lines.append(f"  Teacher notes: {r.teacher_notes}")
            lines.append(f"  Student notes: {r.student_notes}")

    if teacher_wins:
        lines.append("")
        lines.append("-" * 100)
        lines.append("DETAILED: TEACHER (BASELINE) WINS")
        lines.append("-" * 100)
        for name, margin, r in sorted(teacher_wins, key=lambda x: -x[1]):
            lines.append(f"\n[{name}] (margin: +{margin})")
            lines.append(f"  Expected: {r.expected[:60]}{'...' if len(r.expected) > 60 else ''}")
            lines.append(f"  Teacher ({r.teacher_score}): {r.teacher_output[:60]}{'...' if len(r.teacher_output) > 60 else ''}")
            lines.append(f"  Student ({r.student_score}): {r.student_output[:60]}{'...' if len(r.student_output) > 60 else ''}")
            lines.append(f"  Teacher notes: {r.teacher_notes}")
            lines.append(f"  Student notes: {r.student_notes}")

    # Critical distractor tests
    distractor_tests = [
        'passkey_choice_logprob_distractors', 'long_passkey_with_noise',
        'sequence_with_distractors', 'copy_semantic_collision',
        'copy_fewshot_spaces', 'copy_fewshot_commas', 'long_sequence_copy',
    ]

    lines.append("")
    lines.append("-" * 100)
    lines.append("CRITICAL TESTS: DISTRACTOR REJECTION")
    lines.append("-" * 100)

    for name in distractor_tests:
        r = next((res for res in summary.results if res.case_id == name), None)
        if r:
            lines.append(f"\n[{name}]")
            lines.append(f"  Teacher: score={r.teacher_score}, distractor_contam={r.teacher_flags.distractor_contamination}")
            lines.append(f"  Student: score={r.student_score}, distractor_contam={r.student_flags.distractor_contamination}")
            if r.student_score > r.teacher_score:
                lines.append("  >>> DBA WINS")
            elif r.teacher_score > r.student_score:
                lines.append("  >>> BASELINE WINS")
            else:
                lines.append("  >>> TIE")

    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for nuanced behavior analysis."""
    import sys
    from pathlib import Path

    # Find the behavior log
    default_paths = [
        "research/dba/dba_checkpoint_benchmark/baseline_vs_decoupled/20260113_145631/behavior_log.txt",
        "dba_checkpoint_benchmark/baseline_vs_decoupled/20260113_145631/behavior_log.txt",
    ]

    log_path: str | None = None

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        for p in default_paths:
            if Path(p).exists():
                log_path = p
                break

    if not log_path or not Path(log_path).exists():
        print("Usage: python nuanced_eval.py <behavior_log.txt>")
        print("Or place log at one of the default paths")
        sys.exit(1)

    print(f"Analyzing: {log_path}")
    print()

    # Parse and analyze
    raw_results = parse_behavior_log(log_path)
    print(f"Parsed {len(raw_results)} test results")

    summary = run_nuanced_eval(raw_results)
    report = format_summary_report(summary)
    print(report)

    # Write report to file
    output_path = Path(log_path).parent / "nuanced_analysis.txt"
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
