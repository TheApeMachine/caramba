"""Nuanced behavioral evaluation for DBA paper.

Extends the standard eval framework with a multi-level scoring system
that distinguishes between exact matches, content-correct responses,
and various failure modes (distractor contamination, repetition loops).

Scoring system:
  3 = Exact match
  2 = Content correct, minor format additions (prefix/suffix)
  1 = Content present but buried in noise
  0 = Wrong content OR distractor contamination
 -1 = Radical failure (garbage, infinite loop, completely unrelated)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


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
class NuancedMeasurement:
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