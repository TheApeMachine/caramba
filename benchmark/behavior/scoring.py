from __future__ import annotations

import re
from dataclasses import dataclass

from benchmark.behavior.types import ContentConstraint, MatchType


def normalize_text(s: str) -> str:
    # Case-insensitive, whitespace-normalized.
    t = str(s).replace("\r\n", "\n").replace("\r", "\n").strip().lower()
    # Normalize all whitespace runs to single spaces.
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass(frozen=True)
class MatchResult:
    match_type: MatchType
    # [start, end) character offsets in normalized output (only for CONTAINED/EXACT_START checks)
    span: tuple[int, int] | None = None


def classify_match(
    *,
    output: str,
    expected: str,
    prompt: str,
    allow_contained: bool,
    contained_constraints: list[ContentConstraint],
    disallow_contained_if_expected_in_prompt: bool,
) -> MatchResult:
    exp = normalize_text(expected)
    if not exp:
        raise ValueError("Expected answer must be non-empty.")

    out_raw = str(output).strip()
    out_norm = normalize_text(out_raw)
    prompt_norm = normalize_text(prompt)

    # EXACT: full output equals expected OR first non-empty line equals expected.
    if out_norm == exp:
        return MatchResult(MatchType.EXACT, span=(0, len(exp)))
    first_line = normalize_text(out_raw.split("\n", 1)[0])
    if first_line == exp:
        return MatchResult(MatchType.EXACT, span=(0, len(exp)))

    # NONE/CONTAINED logic
    if not allow_contained:
        return MatchResult(MatchType.NONE, span=None)

    if disallow_contained_if_expected_in_prompt and exp in prompt_norm:
        return MatchResult(MatchType.NONE, span=None)

    # Boolean answers: only accept if the *first* boolean token matches expected.
    if exp in {"true", "false", "yes", "no"}:
        m = re.search(r"\b(true|false|yes|no)\b", out_norm)
        if m is None:
            return MatchResult(MatchType.NONE, span=None)
        first_bool = str(m.group(1))
        if first_bool != exp:
            return MatchResult(MatchType.NONE, span=None)
        # Still require constraints (if any) against the full normalized output.
        span = (int(m.start(1)), int(m.end(1)))
        if not _constraints_ok(
            out_norm=out_norm,
            prompt_norm=prompt_norm,
            exp=exp,
            span=span,
            constraints=contained_constraints,
        ):
            return MatchResult(MatchType.NONE, span=None)
        return MatchResult(MatchType.CONTAINED, span=span)

    idx = out_norm.find(exp)
    if idx < 0:
        return MatchResult(MatchType.NONE, span=None)
    span = (int(idx), int(idx) + len(exp))

    if not _constraints_ok(
        out_norm=out_norm,
        prompt_norm=prompt_norm,
        exp=exp,
        span=span,
        constraints=contained_constraints,
    ):
        return MatchResult(MatchType.NONE, span=None)

    return MatchResult(MatchType.CONTAINED, span=span)


def _constraints_ok(
    *,
    out_norm: str,
    prompt_norm: str,
    exp: str,
    span: tuple[int, int],
    constraints: list[ContentConstraint],
) -> bool:
    s0, s1 = span
    if s0 < 0 or s1 < 0 or s1 < s0:
        raise ValueError("Invalid span passed to constraints check.")
    if not constraints:
        return True

    out_len = max(1, len(out_norm))
    match_len = max(1, (s1 - s0))

    for c in constraints:
        if c == ContentConstraint.MAJORITY:
            # matching segment makes up at least 50% of the output
            if (float(match_len) / float(out_len)) < 0.5:
                return False
        elif c == ContentConstraint.EXACT_START:
            if s0 != 0:
                return False
        elif c == ContentConstraint.NO_PARROTING:
            # Strict: prompt + expected must not appear verbatim in the output.
            if (prompt_norm + " " + exp) in out_norm:
                return False
        else:
            raise ValueError(f"Unknown content constraint: {c!r}")
    return True


def match_score(mt: MatchType) -> float:
    if mt == MatchType.EXACT:
        return 1.0
    if mt == MatchType.CONTAINED:
        return 0.5
    if mt == MatchType.NONE:
        return 0.0
    raise ValueError(f"Unknown match type: {mt!r}")

