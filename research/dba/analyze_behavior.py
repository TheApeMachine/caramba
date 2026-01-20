#!/usr/bin/env python3
"""
Nuanced behavioral benchmark analysis for DBA paper.

Scoring system:
  3 = Exact match
  2 = Content correct, minor format additions (prefix/suffix)
  1 = Content present but buried in noise/repetition
  0 = Wrong content OR distractor contamination
 -1 = Radical failure (garbage, infinite loop, completely unrelated)

Additional flags:
  - repetition_loop: Did the model enter a pathological loop?
  - distractor_contamination: Did the model output earlier examples instead of target?
  - format_continuation: Did model continue prompt structure but get right content?
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TestResult:
    name: str
    prompt: str
    expected: str
    teacher_output: str
    student_output: str
    teacher_strict_pass: bool
    student_strict_pass: bool

    # Nuanced scores
    teacher_score: int = 0
    student_score: int = 0

    # Flags
    teacher_repetition_loop: bool = False
    student_repetition_loop: bool = False
    teacher_distractor_contamination: bool = False
    student_distractor_contamination: bool = False
    teacher_format_continuation: bool = False
    student_format_continuation: bool = False

    # Analysis notes
    teacher_notes: str = ""
    student_notes: str = ""


def detect_repetition_loop(output: str, min_repeats: int = 3) -> bool:
    """Detect if output contains pathological repetition."""
    if len(output) < 20:
        return False

    # Check for repeated phrases (3+ word sequences repeated 3+ times)
    words = output.split()
    if len(words) < 9:
        return False

    for phrase_len in range(1, 5):
        for i in range(len(words) - phrase_len * min_repeats):
            phrase = tuple(words[i:i+phrase_len])
            count = 0
            j = i
            while j <= len(words) - phrase_len:
                if tuple(words[j:j+phrase_len]) == phrase:
                    count += 1
                    j += phrase_len
                else:
                    break
            if count >= min_repeats:
                return True

    # Check for repeated characters/tokens
    if re.search(r'(.{2,10})\1{3,}', output):
        return True

    return False


def detect_distractor_contamination(output: str, prompt: str, expected: str) -> bool:
    """Check if output contains content from earlier examples instead of target."""
    # Extract potential distractors from prompt (earlier examples)
    lines = prompt.strip().split('\n')

    # Common patterns for few-shot examples
    distractors = []
    for line in lines[:-2]:  # Exclude the last line (the query)
        # Extract values after common separators
        for pattern in [r'Output:\s*(.+?)\.?\s*$', r'Copy:\s*(.+?)\.?\s*$',
                       r'->\s*(.+?)\.?\s*$', r':\s*(.+?)\.?\s*$']:
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


def extract_content(output: str, expected: str) -> tuple[bool, bool, str]:
    """
    Check if expected content is present in output.
    Returns: (content_present, is_exact, extracted_content)
    """
    output_clean = output.strip()
    expected_clean = expected.strip()

    # Exact match
    if output_clean == expected_clean:
        return True, True, output_clean

    # Content present somewhere in output
    if expected_clean in output_clean:
        return True, False, expected_clean

    # Try case-insensitive
    if expected_clean.lower() in output_clean.lower():
        return True, False, expected_clean

    # Try without punctuation
    output_alphanum = re.sub(r'[^\w\s]', '', output_clean)
    expected_alphanum = re.sub(r'[^\w\s]', '', expected_clean)
    if expected_alphanum and expected_alphanum in output_alphanum:
        return True, False, expected_alphanum

    return False, False, ""


def detect_format_continuation(output: str, prompt: str, expected: str) -> bool:
    """Check if model continued prompt format but got right content."""
    # Common format continuations
    format_prefixes = ['Input:', 'Output:', 'Text:', 'Copy:', 'X ->', 'X:',
                       'Sequence:', 'Pattern:', 'Data:', 'Row:', 'book ->']

    output_clean = output.strip()
    expected_clean = expected.strip()

    # Check if output has format prefix but contains expected content
    has_prefix = any(output_clean.startswith(p) or output_clean.startswith(p.lower())
                     for p in format_prefixes)
    content_present = expected_clean in output_clean or expected_clean.lower() in output_clean.lower()

    return has_prefix and content_present


def score_output(output: str, expected: str, prompt: str) -> tuple[int, str, dict]:
    """
    Score an output on the nuanced scale.
    Returns: (score, notes, flags)
    """
    flags = {
        'repetition_loop': False,
        'distractor_contamination': False,
        'format_continuation': False,
    }

    output_clean = output.strip()
    expected_clean = expected.strip()

    # Check for repetition loop first (radical failure)
    if detect_repetition_loop(output):
        flags['repetition_loop'] = True
        return -1, "Repetition loop detected", flags

    # Check for exact match
    if output_clean == expected_clean:
        return 3, "Exact match", flags

    # Check if content is present
    content_present, is_exact, _ = extract_content(output, expected)

    if content_present:
        # Check if it's format continuation
        if detect_format_continuation(output, prompt, expected):
            flags['format_continuation'] = True
            return 2, "Content correct, format continuation", flags

        # Content present but maybe with extra stuff
        # Check how much extra
        len_ratio = len(output_clean) / max(len(expected_clean), 1)
        if len_ratio < 2.0:
            return 2, "Content correct, minor additions", flags
        else:
            return 1, "Content present but buried", flags

    # Check for distractor contamination
    if detect_distractor_contamination(output, prompt, expected):
        flags['distractor_contamination'] = True
        return 0, "Distractor contamination", flags

    # Check for garbage output
    if len(output_clean) > 100 and not any(c.isalnum() for c in output_clean[:50]):
        return -1, "Garbage output", flags

    # Wrong content
    return 0, "Wrong content", flags


def parse_behavior_log(log_path: str) -> list[TestResult]:
    """Parse the behavior log file into structured results."""
    with open(log_path, 'r') as f:
        content = f.read()

    results = []

    # Split by test delimiter
    test_blocks = re.split(r'\n\[(\d+)/73\]\s+(\w+)\n-+\n', content)

    # Skip header
    i = 1
    while i < len(test_blocks) - 2:
        test_num = test_blocks[i]
        test_name = test_blocks[i + 1]
        test_content = test_blocks[i + 2]

        # Parse test content
        prompt_match = re.search(r'PROMPT:\n(.*?)\n\nEXPECTED:', test_content, re.DOTALL)
        expected_match = re.search(r'EXPECTED:\s*(.*?)\n\nTEACHER', test_content, re.DOTALL)
        teacher_match = re.search(r'TEACHER ([✓✗]):\n(.*?)\n\nSTUDENT', test_content, re.DOTALL)
        student_match = re.search(r'STUDENT ([✓✗]):\n(.*?)(?:\n\n|\n*$)', test_content, re.DOTALL)

        # Type narrowing: check each match individually
        if prompt_match is not None and expected_match is not None and teacher_match is not None and student_match is not None:
            result = TestResult(
                name=test_name,
                prompt=prompt_match.group(1).strip(),
                expected=expected_match.group(1).strip(),
                teacher_output=teacher_match.group(2).strip(),
                student_output=student_match.group(2).strip(),
                teacher_strict_pass=(teacher_match.group(1) == '✓'),
                student_strict_pass=(student_match.group(1) == '✓'),
            )
            results.append(result)

        i += 3

    return results


def analyze_results(results: list[TestResult]) -> None:
    """Perform nuanced analysis and scoring."""
    for result in results:
        # Score teacher
        score, notes, flags = score_output(result.teacher_output, result.expected, result.prompt)
        result.teacher_score = score
        result.teacher_notes = notes
        result.teacher_repetition_loop = flags['repetition_loop']
        result.teacher_distractor_contamination = flags['distractor_contamination']
        result.teacher_format_continuation = flags['format_continuation']

        # Score student
        score, notes, flags = score_output(result.student_output, result.expected, result.prompt)
        result.student_score = score
        result.student_notes = notes
        result.student_repetition_loop = flags['repetition_loop']
        result.student_distractor_contamination = flags['distractor_contamination']
        result.student_format_continuation = flags['format_continuation']


def print_report(results: list[TestResult]) -> None:
    """Print comprehensive analysis report."""
    print("=" * 100)
    print("NUANCED BEHAVIORAL ANALYSIS REPORT")
    print("=" * 100)
    print()

    # Summary statistics
    teacher_scores = [r.teacher_score for r in results]
    student_scores = [r.student_score for r in results]

    print("SCORING LEGEND:")
    print("  3 = Exact match")
    print("  2 = Content correct, minor format additions")
    print("  1 = Content present but buried")
    print("  0 = Wrong content / distractor contamination")
    print(" -1 = Radical failure (loop, garbage)")
    print()

    print("-" * 100)
    print("OVERALL SCORES")
    print("-" * 100)
    print(f"{'Metric':<40} {'Teacher (Baseline)':<25} {'Student (DBA)':<25}")
    print("-" * 100)

    # Strict pass rate
    teacher_strict = sum(1 for r in results if r.teacher_strict_pass)
    student_strict = sum(1 for r in results if r.student_strict_pass)
    print(f"{'Strict Pass (exact match)':<40} {teacher_strict}/73 ({100*teacher_strict/73:.1f}%){'':<10} {student_strict}/73 ({100*student_strict/73:.1f}%)")

    # Score >= 2 (content correct)
    teacher_content = sum(1 for r in results if r.teacher_score >= 2)
    student_content = sum(1 for r in results if r.student_score >= 2)
    print(f"{'Content Correct (score >= 2)':<40} {teacher_content}/73 ({100*teacher_content/73:.1f}%){'':<10} {student_content}/73 ({100*student_content/73:.1f}%)")

    # Score >= 1 (content present)
    teacher_present = sum(1 for r in results if r.teacher_score >= 1)
    student_present = sum(1 for r in results if r.student_score >= 1)
    print(f"{'Content Present (score >= 1)':<40} {teacher_present}/73 ({100*teacher_present/73:.1f}%){'':<10} {student_present}/73 ({100*student_present/73:.1f}%)")

    # Total score
    teacher_total = sum(teacher_scores)
    student_total = sum(student_scores)
    print(f"{'Total Score':<40} {teacher_total}{'':<20} {student_total}")

    # Average score
    teacher_avg = sum(teacher_scores) / len(teacher_scores)
    student_avg = sum(student_scores) / len(student_scores)
    print(f"{'Average Score':<40} {teacher_avg:.2f}{'':<20} {student_avg:.2f}")

    print()
    print("-" * 100)
    print("FAILURE MODE ANALYSIS")
    print("-" * 100)

    # Repetition loops
    teacher_loops = sum(1 for r in results if r.teacher_repetition_loop)
    student_loops = sum(1 for r in results if r.student_repetition_loop)
    print(f"{'Repetition Loops':<40} {teacher_loops}{'':<20} {student_loops}")

    # Distractor contamination
    teacher_distract = sum(1 for r in results if r.teacher_distractor_contamination)
    student_distract = sum(1 for r in results if r.student_distractor_contamination)
    print(f"{'Distractor Contamination':<40} {teacher_distract}{'':<20} {student_distract}")

    # Format continuation (not a failure, but interesting)
    teacher_format = sum(1 for r in results if r.teacher_format_continuation)
    student_format = sum(1 for r in results if r.student_format_continuation)
    print(f"{'Format Continuation (content correct)':<40} {teacher_format}{'':<20} {student_format}")

    print()
    print("-" * 100)
    print("HEAD-TO-HEAD COMPARISON")
    print("-" * 100)

    # Cases where student beat teacher
    student_wins = [(r.name, r.student_score - r.teacher_score) for r in results if r.student_score > r.teacher_score]
    teacher_wins = [(r.name, r.teacher_score - r.student_score) for r in results if r.teacher_score > r.student_score]
    ties = [r.name for r in results if r.teacher_score == r.student_score]

    print(f"Student (DBA) wins: {len(student_wins)}")
    print(f"Teacher (Baseline) wins: {len(teacher_wins)}")
    print(f"Ties: {len(ties)}")

    print()
    print("-" * 100)
    print("DETAILED: STUDENT (DBA) WINS")
    print("-" * 100)
    for name, margin in sorted(student_wins, key=lambda x: -x[1]):
        r = next(res for res in results if res.name == name)
        print(f"\n[{name}] (margin: +{margin})")
        print(f"  Expected: {r.expected[:60]}{'...' if len(r.expected) > 60 else ''}")
        print(f"  Teacher ({r.teacher_score}): {r.teacher_output[:60]}{'...' if len(r.teacher_output) > 60 else ''}")
        print(f"  Student ({r.student_score}): {r.student_output[:60]}{'...' if len(r.student_output) > 60 else ''}")
        print(f"  Teacher notes: {r.teacher_notes}")
        print(f"  Student notes: {r.student_notes}")

    print()
    print("-" * 100)
    print("DETAILED: TEACHER (BASELINE) WINS")
    print("-" * 100)
    for name, margin in sorted(teacher_wins, key=lambda x: -x[1]):
        r = next(res for res in results if res.name == name)
        print(f"\n[{name}] (margin: +{margin})")
        print(f"  Expected: {r.expected[:60]}{'...' if len(r.expected) > 60 else ''}")
        print(f"  Teacher ({r.teacher_score}): {r.teacher_output[:60]}{'...' if len(r.teacher_output) > 60 else ''}")
        print(f"  Student ({r.student_score}): {r.student_output[:60]}{'...' if len(r.student_output) > 60 else ''}")
        print(f"  Teacher notes: {r.teacher_notes}")
        print(f"  Student notes: {r.student_notes}")

    print()
    print("-" * 100)
    print("CRITICAL TESTS: DISTRACTOR REJECTION")
    print("-" * 100)
    distractor_tests = ['passkey_choice_logprob_distractors', 'long_passkey_with_noise',
                        'sequence_with_distractors', 'copy_semantic_collision',
                        'copy_fewshot_spaces', 'copy_fewshot_commas', 'long_sequence_copy']
    for name in distractor_tests:
        r = next((res for res in results if res.name == name), None)
        if r:
            print(f"\n[{name}]")
            print(f"  Teacher: score={r.teacher_score}, distractor_contam={r.teacher_distractor_contamination}")
            print(f"  Student: score={r.student_score}, distractor_contam={r.student_distractor_contamination}")
            if r.student_score > r.teacher_score:
                print(f"  >>> DBA WINS")
            elif r.teacher_score > r.student_score:
                print(f"  >>> BASELINE WINS")
            else:
                print(f"  >>> TIE")

    print()
    print("=" * 100)
    print("END OF REPORT")
    print("=" * 100)


def main():
    import sys

    # Find the behavior log
    log_paths = [
        "research/dba/dba_checkpoint_benchmark/baseline_vs_decoupled/20260113_145631/behavior_log.txt",
        "/sessions/determined-epic-brahmagupta/mnt/dba/behavior_log.txt",
    ]

    log_path = None
    for p in log_paths:
        if Path(p).exists():
            log_path = p
            break

    if not log_path and len(sys.argv) > 1:
        log_path = sys.argv[1]

    if not log_path or not Path(log_path).exists():
        print("Usage: python analyze_behavior.py <behavior_log.txt>")
        print("Or place log at one of the default paths")
        sys.exit(1)

    print(f"Analyzing: {log_path}")
    print()

    results = parse_behavior_log(log_path)
    print(f"Parsed {len(results)} test results")

    analyze_results(results)
    print_report(results)


if __name__ == "__main__":
    main()
