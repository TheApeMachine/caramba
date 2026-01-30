#!/usr/bin/env python3
"""Export the *fully materialized* unified behavior suite to CSV.

This expands the dynamic template suite (slot-based YAML) into concrete test cases.

Default suite:
  benchmark/behavior/cases.yml

Example:
  python scripts/export_behavior_cases_csv.py \
    --out benchmark/behavior/questions_expected.csv

Notes:
- By default, this **escapes newlines** in prompt/expected/target_text as literal "\\n"
  so each CSV record stays on a single line (easy to diff/grep). Disable with
  `--no-escape-newlines` if you prefer raw multiline fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Allow running this script directly (so it can import Caramba modules).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.behavior.suite import load_behavior_suite  # noqa: E402


def _escape_newlines(s: str | None) -> str:
    if s is None:
        return ""
    # Keep backslashes stable for round-tripping.
    return str(s).replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export fully realized behavior cases to CSV.")
    ap.add_argument(
        "--suite",
        type=str,
        default="benchmark/behavior/cases.yml",
        help="Path to behavior suite YAML (default: benchmark/behavior/cases.yml)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="-",
        help="Output CSV path, or '-' for stdout (default: '-')",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override (default: use suite seed)",
    )
    ap.add_argument(
        "--no-escape-newlines",
        action="store_true",
        help="Do not escape newlines in prompt/expected/target_text (raw multiline CSV fields).",
    )
    ap.add_argument(
        "--include-choices",
        action="store_true",
        help="Include choice fields (choices_json, correct_index) for choice_logprob cases.",
    )

    args = ap.parse_args(argv)

    suite_path = Path(args.suite)
    _spec, cases = load_behavior_suite(suite_path, seed_override=args.seed)

    escape = not bool(args.no_escape_newlines)
    esc = _escape_newlines if escape else (lambda x: "" if x is None else str(x))

    fieldnames = ["task_type", "category", "difficulty", "kind", "prompt", "expected"]
    if args.include_choices:
        fieldnames += ["choices_json", "correct_index"]
    fieldnames += ["target_text"]

    out: Any
    close_out = False
    if str(args.out) == "-":
        out = None  # handled by newline=''
        f = getattr(__import__("sys"), "stdout")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(out_path, "w", encoding="utf-8", newline="")
        close_out = True

    try:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()

        for c in cases:
            row: dict[str, Any] = {
                "task_type": str(c.id),
                "category": str(c.category),
                "difficulty": getattr(c.difficulty, "value", str(c.difficulty)),
                "kind": getattr(c.kind, "value", str(c.kind)),
                "prompt": esc(c.prompt),
                "expected": esc(c.expected),
                "target_text": esc(c.target_text),
            }
            if args.include_choices:
                row["choices_json"] = json.dumps(list(c.choices or []), ensure_ascii=False)
                row["correct_index"] = "" if c.correct_index is None else int(c.correct_index)

            w.writerow(row)
    finally:
        if close_out:
            f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

