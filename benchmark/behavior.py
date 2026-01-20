"""Behavior benchmark: prompt-suite checks for teacher/student.

This is a lightweight "unit test" style benchmark that runs a curated set of
prompt cases with ground-truth evaluation logic and produces:
- per-case outcomes (teacher_ok/student_ok + extracted answers)
- aggregate accuracies
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast, Protocol

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
import shutil

from config.benchmark import BehaviorBenchmarkConfig
from config.eval import EvalCase, EvalThresholds, EvalVerifyConfig, TokenizerConfig
from console import logger

# Import weighted scoring and visualization
from research.dba.behavioral_suite_v2.weighted_scoring import (
    MatchType,
    MATCH_SCORES,
    DIFFICULTY_WEIGHTS,
    DIFFICULTY_NAMES,
    WeightedModelSummary,
    DifficultyBreakdown,
    CategoryBreakdown,
    WeightedScorer,
    classify_match_type,
)
from research.dba.behavioral_suite_v2.weighted_visualizer import (
    generate_all_weighted_visualizations,
    generate_latex_ranking_table,
)

from eval.suite import run_eval_verify
from eval.nuanced import score_output, NuancedFlags
from data.tokenizers.builder import TokenizerBuilder
from instrumentation.viz import TrainingVizContext
from caramba.layer.attention import AttentionLayer
from benchmark.attention_multi_model import (
    render_multi_model_attention_comparison,
    MultiModelAttentionVisualizer,
)


class ModelWithCtx(Protocol):
    def __call__(self, x: torch.Tensor, ctx: Any = None) -> Any: ...


@dataclass
class BehaviorMeasurement:
    case_id: str
    teacher_ok: bool
    student_ok: bool
    teacher_answer: str
    student_answer: str

    # Nuanced scoring (optional, for enhanced analysis)
    teacher_score: int | None = None  # NuancedScore enum value
    student_score: int | None = None  # NuancedScore enum value
    teacher_flags: dict | None = None  # Diagnostic flags
    student_flags: dict | None = None  # Diagnostic flags
    teacher_notes: str = ""
    student_notes: str = ""

    # Weighted scoring (hard/soft with difficulty weighting)
    expected_answer: str = ""  # For weighted scoring reference
    teacher_match_type: MatchType | None = None  # NONE, CONTAINED, EXACT
    student_match_type: MatchType | None = None  # NONE, CONTAINED, EXACT
    teacher_raw_score: float = 0.0  # 0.0, 0.5, or 1.0
    student_raw_score: float = 0.0  # 0.0, 0.5, or 1.0
    difficulty_weight: float = 1.0  # 1.0 (easy), 2.0 (medium), 3.0 (hard)
    student_weighted_score: float = 0.0  # student_raw_score × difficulty_weight


@dataclass
class BehaviorResult:
    """Behavior benchmark results for a teacher/student pair."""

    benchmark_id: str
    measurements: list[BehaviorMeasurement] = field(default_factory=list)

    @property
    def teacher_accuracy(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(1 for m in self.measurements if m.teacher_ok) / float(len(self.measurements))

    @property
    def student_accuracy(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(1 for m in self.measurements if m.student_ok) / float(len(self.measurements))

    # Nuanced scoring metrics
    @property
    def teacher_content_accuracy(self) -> float:
        """Fraction of cases where teacher score >= 2 (content correct)."""
        valid_measurements = [m for m in self.measurements if m.teacher_score is not None]
        if not valid_measurements:
            return 0.0
        return sum(1 for m in valid_measurements if m.teacher_score is not None and m.teacher_score >= 2) / float(len(valid_measurements))

    @property
    def student_content_accuracy(self) -> float:
        """Fraction of cases where student score >= 2 (content correct)."""
        valid_measurements = [m for m in self.measurements if m.student_score is not None]
        if not valid_measurements:
            return 0.0
        return sum(1 for m in valid_measurements if m.student_score is not None and m.student_score >= 2) / float(len(valid_measurements))

    @property
    def teacher_avg_score(self) -> float:
        """Average nuanced score for teacher."""
        valid_scores = [m.teacher_score for m in self.measurements if m.teacher_score is not None]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / float(len(valid_scores))

    @property
    def student_avg_score(self) -> float:
        """Average nuanced score for student."""
        valid_scores = [m.student_score for m in self.measurements if m.student_score is not None]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / float(len(valid_scores))

    @property
    def teacher_repetition_loops(self) -> int:
        """Number of cases where teacher exhibited repetition loops."""
        return sum(1 for m in self.measurements
                  if m.teacher_flags and m.teacher_flags.get("repetition_loop", False))

    @property
    def student_repetition_loops(self) -> int:
        """Number of cases where student exhibited repetition loops."""
        return sum(1 for m in self.measurements
                  if m.student_flags and m.student_flags.get("repetition_loop", False))

    @property
    def teacher_distractor_contaminations(self) -> int:
        """Number of cases where teacher output contained distractors."""
        return sum(1 for m in self.measurements
                  if m.teacher_flags and m.teacher_flags.get("distractor_contamination", False))

    @property
    def student_distractor_contaminations(self) -> int:
        """Number of cases where student output contained distractors."""
        return sum(1 for m in self.measurements
                  if m.student_flags and m.student_flags.get("distractor_contamination", False))

    # Weighted scoring metrics (hard/soft)
    @property
    def teacher_hard_accuracy(self) -> float:
        """Fraction of cases where teacher got EXACT match."""
        valid = [m for m in self.measurements if m.teacher_match_type is not None]
        if not valid:
            return 0.0
        return sum(1 for m in valid if m.teacher_match_type == MatchType.EXACT) / len(valid)

    @property
    def teacher_soft_accuracy(self) -> float:
        """Fraction of cases where teacher got EXACT or CONTAINED match."""
        valid = [m for m in self.measurements if m.teacher_match_type is not None]
        if not valid:
            return 0.0
        return sum(1 for m in valid if m.teacher_match_type in (MatchType.EXACT, MatchType.CONTAINED)) / len(valid)

    @property
    def student_hard_accuracy(self) -> float:
        """Fraction of cases where student got EXACT match."""
        valid = [m for m in self.measurements if m.student_match_type is not None]
        if not valid:
            return 0.0
        return sum(1 for m in valid if m.student_match_type == MatchType.EXACT) / len(valid)

    @property
    def student_soft_accuracy(self) -> float:
        """Fraction of cases where student got EXACT or CONTAINED match."""
        valid = [m for m in self.measurements if m.student_match_type is not None]
        if not valid:
            return 0.0
        return sum(1 for m in valid if m.student_match_type in (MatchType.EXACT, MatchType.CONTAINED)) / len(valid)

    @property
    def student_weighted_accuracy(self) -> float:
        """Student weighted accuracy (weighted_score_sum / max_weighted_score_sum)."""
        valid = [m for m in self.measurements if m.teacher_match_type is not None]
        if not valid:
            return 0.0
        weighted_sum = sum(m.student_weighted_score for m in valid)
        max_weighted_sum = sum(m.difficulty_weight for m in valid)  # max score = 1.0 × weight
        return weighted_sum / max_weighted_sum if max_weighted_sum > 0 else 0.0

    @property
    def weighted_score_summary(self) -> dict[str, Any]:
        """Summary of weighted scoring metrics."""
        valid = [m for m in self.measurements if m.teacher_match_type is not None]
        if not valid:
            return {
                "total_tests": 0,
                "teacher_hard_accuracy": 0.0,
                "teacher_soft_accuracy": 0.0,
                "student_hard_accuracy": 0.0,
                "student_soft_accuracy": 0.0,
                "student_weighted_accuracy": 0.0,
            }

        n = len(valid)
        t_exact = sum(1 for m in valid if m.teacher_match_type == MatchType.EXACT)
        t_contained = sum(1 for m in valid if m.teacher_match_type == MatchType.CONTAINED)
        s_exact = sum(1 for m in valid if m.student_match_type == MatchType.EXACT)
        s_contained = sum(1 for m in valid if m.student_match_type == MatchType.CONTAINED)

        weighted_sum = sum(m.student_weighted_score for m in valid)
        max_weighted_sum = sum(m.difficulty_weight for m in valid)

        # Breakdown by difficulty
        easy = [m for m in valid if m.difficulty_weight == 1.0]
        medium = [m for m in valid if m.difficulty_weight == 2.0]
        hard = [m for m in valid if m.difficulty_weight == 3.0]

        return {
            "total_tests": n,
            "teacher_exact_count": t_exact,
            "teacher_contained_count": t_contained,
            "teacher_none_count": n - t_exact - t_contained,
            "student_exact_count": s_exact,
            "student_contained_count": s_contained,
            "student_none_count": n - s_exact - s_contained,
            "teacher_hard_accuracy": t_exact / n,
            "teacher_soft_accuracy": (t_exact + t_contained) / n,
            "student_hard_accuracy": s_exact / n,
            "student_soft_accuracy": (s_exact + s_contained) / n,
            "student_weighted_score_sum": weighted_sum,
            "student_weighted_score_max": max_weighted_sum,
            "student_weighted_accuracy": weighted_sum / max_weighted_sum if max_weighted_sum > 0 else 0.0,
            "difficulty_breakdown": {
                "easy": {"count": len(easy), "student_exact": sum(1 for m in easy if m.student_match_type == MatchType.EXACT)},
                "medium": {"count": len(medium), "student_exact": sum(1 for m in medium if m.student_match_type == MatchType.EXACT)},
                "hard": {"count": len(hard), "student_exact": sum(1 for m in hard if m.student_match_type == MatchType.EXACT)},
            },
        }


@dataclass
class BehaviorMultiMeasurement:
    """Measurement for a single case across N models."""
    case_id: str
    expected: str
    prompt: str = ""
    category: str = ""

    # Per-model results: {model_name: output_str}
    model_outputs: dict[str, str] = field(default_factory=dict)

    # Per-model match types: {model_name: MatchType}
    model_match_types: dict[str, MatchType] = field(default_factory=dict)

    # Per-model raw scores: {model_name: float (0.0, 0.5, 1.0)}
    model_raw_scores: dict[str, float] = field(default_factory=dict)

    # Difficulty weight (based on baseline performance)
    difficulty_weight: float = 1.0

    # Per-model weighted scores: {model_name: weighted_score}
    model_weighted_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BehaviorMultiResult:
    """Results for N-model behavior benchmark with weighted scoring."""
    benchmark_id: str
    baseline_name: str = ""
    measurements: list[BehaviorMultiMeasurement] = field(default_factory=list)

    # Model names in order
    model_names: list[str] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.measurements)

    def get_hard_accuracy(self, model_name: str) -> float:
        """Fraction of EXACT matches for a model."""
        valid = [m for m in self.measurements if model_name in m.model_match_types]
        if not valid:
            return 0.0
        exact = sum(1 for m in valid if m.model_match_types.get(model_name) == MatchType.EXACT)
        return exact / len(valid)

    def get_soft_accuracy(self, model_name: str) -> float:
        """Fraction of EXACT or CONTAINED matches for a model."""
        valid = [m for m in self.measurements if model_name in m.model_match_types]
        if not valid:
            return 0.0
        ok = sum(1 for m in valid if m.model_match_types.get(model_name) in (MatchType.EXACT, MatchType.CONTAINED))
        return ok / len(valid)

    def get_weighted_accuracy(self, model_name: str) -> float:
        """Weighted accuracy for a model (sum of weighted_scores / max possible)."""
        valid = [m for m in self.measurements if model_name in m.model_weighted_scores]
        if not valid:
            return 0.0
        weighted_sum = sum(m.model_weighted_scores.get(model_name, 0.0) for m in valid)
        max_sum = sum(m.difficulty_weight for m in valid)  # max score = 1.0 × weight
        return weighted_sum / max_sum if max_sum > 0 else 0.0

    def get_weighted_summary(self) -> dict[str, Any]:
        """Get comprehensive weighted scoring summary."""
        summary: dict[str, Any] = {
            "total_tests": self.total_tests,
            "baseline_name": self.baseline_name,
            "models": {},
        }

        # Difficulty distribution
        easy = sum(1 for m in self.measurements if m.difficulty_weight == 1.0)
        medium = sum(1 for m in self.measurements if m.difficulty_weight == 2.0)
        hard = sum(1 for m in self.measurements if m.difficulty_weight == 3.0)
        summary["difficulty_distribution"] = {"easy": easy, "medium": medium, "hard": hard}

        for model_name in self.model_names:
            valid = [m for m in self.measurements if model_name in m.model_match_types]
            n = len(valid)
            if n == 0:
                continue

            exact = sum(1 for m in valid if m.model_match_types.get(model_name) == MatchType.EXACT)
            contained = sum(1 for m in valid if m.model_match_types.get(model_name) == MatchType.CONTAINED)
            none_count = n - exact - contained

            weighted_sum = sum(m.model_weighted_scores.get(model_name, 0.0) for m in valid)
            max_sum = sum(m.difficulty_weight for m in valid)

            summary["models"][model_name] = {
                "exact_count": exact,
                "contained_count": contained,
                "none_count": none_count,
                "hard_accuracy": exact / n,
                "soft_accuracy": (exact + contained) / n,
                "weighted_score_sum": weighted_sum,
                "weighted_score_max": max_sum,
                "weighted_accuracy": weighted_sum / max_sum if max_sum > 0 else 0.0,
            }

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "baseline_name": self.baseline_name,
            "model_names": self.model_names,
            "weighted_summary": self.get_weighted_summary(),
            "measurements": [
                {
                    "case_id": m.case_id,
                    "prompt": m.prompt,
                    "expected": m.expected,
                    "model_outputs": m.model_outputs,
                    "difficulty_weight": m.difficulty_weight,
                    "model_match_types": {k: v.name for k, v in m.model_match_types.items()},
                    "model_raw_scores": m.model_raw_scores,
                    "model_weighted_scores": m.model_weighted_scores,
                }
                for m in self.measurements
            ],
        }


class BehaviorBenchmark:
    def __init__(self, config: BehaviorBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def _load_cases(self) -> list[EvalCase]:
        path = Path(self.config.cases_file)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("BehaviorBenchmark cases_file must be a non-empty YAML list")
        cases: list[EvalCase] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(
                    f"BehaviorBenchmark cases must be dict objects, got {type(item)!r} at index {i}"
                )
            cases.append(EvalCase.model_validate(item))
        return cases

    def run(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        benchmark_id: str,
        output_dir: Path | None = None,
    ) -> BehaviorResult:
        stream_live = bool(getattr(self.config, "stream_live", False))
        max_chars = int(getattr(self.config, "print_max_chars", 160))
        log_file = getattr(self.config, "log_file", None)

        def _truncate(s: str, mx: int = max_chars) -> str:
            ss = str(s).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            return ss if len(ss) <= mx else ss[: max(0, mx - 1)] + "…"

        # Leverage the existing eval suite implementation.
        cfg = EvalVerifyConfig(
            tokenizer=self.config.tokenizer,
            max_new_tokens=int(self.config.max_new_tokens),
            context_window=self.config.context_window,
            # thresholds are irrelevant for benchmarking (we always record), but required by config
            thresholds=EvalThresholds(min_student_accuracy=0.0, max_accuracy_drop=1.0),
            cases=self._load_cases(),
        )
        # Keep cases for prompt lookup during streaming
        cases = cfg.cases
        case_by_id = {str(c.id): c for c in cases}

        summary = run_eval_verify(
            teacher=teacher,
            student=student,
            cfg=cfg,
            device=self.device,
        )
        out = BehaviorResult(benchmark_id=str(benchmark_id))

        if stream_live:
            logger.console.print()
            logger.console.print(f"[highlight]━━━ Behavior Benchmark: {benchmark_id} ━━━[/highlight]")
            logger.console.print()

        # Prepare log file if configured
        log_lines: list[str] = []
        if log_file:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"BEHAVIOR BENCHMARK: {benchmark_id}")
            log_lines.append(f"{'=' * 80}")
            log_lines.append("")

        for i, r in enumerate(summary.results):
            m = BehaviorMeasurement(
                case_id=str(r.case_id),
                teacher_ok=bool(r.teacher_ok),
                student_ok=bool(r.student_ok),
                teacher_answer=str(r.teacher_answer),
                student_answer=str(r.student_answer),
            )

            # Get the original prompt and expected answer
            case = case_by_id.get(m.case_id)

            # Add nuanced scoring if we have the prompt and expected answer
            if case:
                # Score teacher output
                t_score, t_notes, t_flags = score_output(
                    m.teacher_answer, str(case.answer), case.prompt
                )
                m.teacher_score = int(t_score)
                m.teacher_notes = t_notes
                m.teacher_flags = {
                    "repetition_loop": t_flags.repetition_loop,
                    "distractor_contamination": t_flags.distractor_contamination,
                    "format_continuation": t_flags.format_continuation,
                }

                # Score student output
                s_score, s_notes, s_flags = score_output(
                    m.student_answer, str(case.answer), case.prompt
                )
                m.student_score = int(s_score)
                m.student_notes = s_notes
                m.student_flags = {
                    "repetition_loop": s_flags.repetition_loop,
                    "distractor_contamination": s_flags.distractor_contamination,
                    "format_continuation": s_flags.format_continuation,
                }

                # Weighted scoring (hard/soft with difficulty weighting)
                expected = str(case.answer)
                m.expected_answer = expected
                m.teacher_match_type = classify_match_type(m.teacher_answer, expected)
                m.student_match_type = classify_match_type(m.student_answer, expected)
                m.teacher_raw_score = MATCH_SCORES[m.teacher_match_type]
                m.student_raw_score = MATCH_SCORES[m.student_match_type]
                # Difficulty weight: this should come from case difficulty, not match type
                m.difficulty_weight = DIFFICULTY_WEIGHTS.get(m.teacher_match_type, 1.0)
                m.student_weighted_score = m.student_raw_score * m.difficulty_weight

            out.measurements.append(m)
            prompt_full = case.prompt if case else "[prompt not found]"
            expected_full = str(case.answer) if case else "?"

            # Live streaming: print each case as it's processed (truncated)
            if stream_live:
                t_status = "[success]✓[/success]" if m.teacher_ok else "[error]✗[/error]"
                s_status = "[success]✓[/success]" if m.student_ok else "[error]✗[/error]"

                logger.console.print(
                    f"  [{i + 1:>2}/{len(summary.results)}] [highlight]{m.case_id}[/highlight]"
                )
                logger.console.print(
                    f"       [muted]prompt:[/muted] {_truncate(prompt_full, 120)}"
                )
                logger.console.print(
                    f"       [muted]expected:[/muted] {_truncate(expected_full, 80)}"
                )
                logger.console.print(
                    f"       {t_status} [muted]teacher:[/muted] {_truncate(m.teacher_answer, 80)}"
                )
                logger.console.print(
                    f"       {s_status} [muted]student:[/muted] {_truncate(m.student_answer, 80)}"
                )
                logger.console.print()

            # Log file: write full untruncated details
            if log_file:
                t_mark = "✓" if m.teacher_ok else "✗"
                s_mark = "✓" if m.student_ok else "✗"
                log_lines.append(f"[{i + 1}/{len(summary.results)}] {m.case_id}")
                log_lines.append("-" * 40)
                log_lines.append("PROMPT:")
                log_lines.append(prompt_full)
                log_lines.append("")
                log_lines.append(f"EXPECTED: {expected_full}")
                log_lines.append("")
                log_lines.append(f"TEACHER {t_mark}:")
                log_lines.append(m.teacher_answer)
                log_lines.append("")
                log_lines.append(f"STUDENT {s_mark}:")
                log_lines.append(m.student_answer)
                log_lines.append("")
                log_lines.append("")

        # Optional attention dump for selected cases (paper/debug).
        if bool(getattr(self.config, "dump_attention", False)):
            try:
                self._dump_attention(
                    teacher=teacher,
                    student=student,
                    cases=cases,
                    measurements=out.measurements,
                    benchmark_id=str(benchmark_id),
                    output_dir=output_dir,
                )
            except Exception as e:
                logger.warning(f"Attention dump failed (continuing): {e!r}")

        # Print summary
        t_correct = sum(1 for m in out.measurements if m.teacher_ok)
        s_correct = sum(1 for m in out.measurements if m.student_ok)
        total = len(out.measurements)

        if stream_live:
            logger.console.print(
                f"[highlight]━━━ Summary:[/highlight] "
                f"teacher [metric]{t_correct}[/metric]/[metric]{total}[/metric] "
                f"([success]{t_correct / total * 100:.1f}%[/success]) │ "
                f"student [metric]{s_correct}[/metric]/[metric]{total}[/metric] "
                f"([success]{s_correct / total * 100:.1f}%[/success])"
            )
            # Print weighted scoring summary if available
            ws = out.weighted_score_summary
            if ws.get("total_tests", 0) > 0:
                logger.console.print(
                    f"[highlight]━━━ Weighted Scores:[/highlight] "
                    f"hard [metric]{ws['student_hard_accuracy'] * 100:.1f}%[/metric] │ "
                    f"soft [metric]{ws['student_soft_accuracy'] * 100:.1f}%[/metric] │ "
                    f"weighted [metric]{ws['student_weighted_accuracy'] * 100:.1f}%[/metric]"
                )
            logger.console.print()

        # Write log file
        if log_file and log_lines:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"SUMMARY: teacher {t_correct}/{total} ({t_correct/total*100:.1f}%) │ student {s_correct}/{total} ({s_correct/total*100:.1f}%)")
            # Add weighted scoring to log
            ws = out.weighted_score_summary
            if ws.get("total_tests", 0) > 0:
                log_lines.append(f"WEIGHTED: hard={ws['student_hard_accuracy']*100:.1f}% │ soft={ws['student_soft_accuracy']*100:.1f}% │ weighted={ws['student_weighted_accuracy']*100:.1f}%")
                log_lines.append(f"  Teacher: EXACT={ws['teacher_exact_count']} CONTAINED={ws['teacher_contained_count']} NONE={ws['teacher_none_count']}")
                log_lines.append(f"  Student: EXACT={ws['student_exact_count']} CONTAINED={ws['student_contained_count']} NONE={ws['student_none_count']}")
            log_lines.append(f"{'=' * 80}")

            log_path = Path(log_file)
            if output_dir and not log_path.is_absolute():
                log_path = output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            logger.info(f"Full behavior log written to: {log_path}")

        # Optional: print per-case outputs for insight/debugging.
        if bool(getattr(self.config, "print_outputs", False)) and out.measurements:
            max_chars = int(getattr(self.config, "print_max_chars", 160))
            only_fail = bool(getattr(self.config, "print_only_failures", True))

            def _trunc(s: str) -> str:
                ss = str(s).replace("\r\n", "\n").replace("\r", "\n")
                ss = ss.replace("\n", "\\n")
                if len(ss) <= max_chars:
                    return ss
                return ss[: max(0, max_chars - 1)] + "…"

            rows: list[list[str]] = []
            for m in out.measurements:
                disagree = bool(m.teacher_ok) != bool(m.student_ok)
                any_wrong = (not bool(m.teacher_ok)) or (not bool(m.student_ok))
                if only_fail and not (disagree or any_wrong):
                    continue
                rows.append(
                    [
                        str(m.case_id),
                        "✓" if bool(m.teacher_ok) else "✗",
                        "✓" if bool(m.student_ok) else "✗",
                        _trunc(m.teacher_answer),
                        _trunc(m.student_answer),
                    ]
                )
            if rows:
                logger.table(
                    title=f"Behavior outputs • {benchmark_id}",
                    columns=["case", "T", "S", "teacher_output", "student_output"],
                    rows=rows,
                )

        return out

    def run_multi(
        self,
        *,
        models: dict[str, nn.Module],
        benchmark_id: str,
        output_dir: Path | None = None,
        baseline_name: str | None = None,
    ) -> BehaviorMultiResult:
        """Run behavior benchmark on N models with weighted scoring.

        Args:
            models: Dict mapping model names to nn.Module instances
            benchmark_id: Identifier for this benchmark run
            output_dir: Optional directory to save results
            baseline_name: Name of baseline model for difficulty weighting (default: first model)

        Returns:
            BehaviorMultiResult with per-model weighted scores
        """
        from data.tokenizers.builder import TokenizerBuilder

        stream_live = bool(getattr(self.config, "stream_live", False))
        max_chars = int(getattr(self.config, "print_max_chars", 160))
        log_file = getattr(self.config, "log_file", None)

        model_names = list(models.keys())
        if baseline_name is None and model_names:
            baseline_name = model_names[0]

        # Set all models to eval mode
        for model in models.values():
            model.eval()

        # Load test cases
        cases = self._load_cases()
        case_by_id = {str(c.id): c for c in cases}

        # Build tokenizer
        tok = TokenizerBuilder().build(self.config.tokenizer)
        # Determine tokenizer vocab size so we can mask padded model logits.
        tok_vocab: int | None = None
        for attr in ("n_vocab", "vocab_size"):
            if hasattr(tok, attr):
                try:
                    tok_vocab = int(getattr(tok, attr))
                    break
                except Exception:
                    tok_vocab = None
        if tok_vocab is None and hasattr(tok, "_enc"):
            try:
                tok_vocab = int(getattr(getattr(tok, "_enc"), "n_vocab"))
            except Exception:
                tok_vocab = None

        # Create result
        out = BehaviorMultiResult(
            benchmark_id=str(benchmark_id),
            baseline_name=baseline_name or model_names[0],
            model_names=model_names,
        )

        if stream_live:
            logger.console.print()
            logger.console.print(f"[highlight]━━━ Behavior Benchmark (Multi): {benchmark_id} ━━━[/highlight]")
            logger.console.print(f"[muted]Models: {', '.join(model_names)} | Baseline: {baseline_name}[/muted]")
            logger.console.print()

        # Run each case
        for i, case in enumerate(cases):
            cid = str(case.id)
            expected = str(case.answer)
            prompt = str(case.prompt)

            # Create measurement
            m = BehaviorMultiMeasurement(
                case_id=cid,
                expected=expected,
                prompt=prompt,
            )

            # Tokenize prompt
            prompt_ids = tok.encode(prompt)

            # Run each model
            for model_name, model in models.items():
                with torch.no_grad():
                    # Generate output (simple greedy)
                    input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
                    for _ in range(int(self.config.max_new_tokens)):
                        logits = model(input_ids)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        last = logits[:, -1, :]
                        if tok_vocab is not None and int(last.shape[-1]) > int(tok_vocab):
                            last = last[..., : int(tok_vocab)]
                        next_token = last.argmax(dim=-1, keepdim=True)
                        if next_token.item() == 0:  # EOS
                            break
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        if self.config.context_window and input_ids.shape[1] > self.config.context_window:
                            break

                    # Decode output
                    generated_ids = input_ids[0, len(prompt_ids):].tolist()
                    output = tok.decode(generated_ids)

                m.model_outputs[model_name] = output

                # Classify match type
                match_type = classify_match_type(output, expected)
                m.model_match_types[model_name] = match_type
                m.model_raw_scores[model_name] = MATCH_SCORES[match_type]

            # Set difficulty weight based on baseline performance
            baseline_match = m.model_match_types.get(baseline_name or "", MatchType.NONE)
            m.difficulty_weight = DIFFICULTY_WEIGHTS.get(baseline_match, 1.0)

            # Compute weighted scores for all models
            for model_name in model_names:
                raw_score = m.model_raw_scores.get(model_name, 0.0)
                m.model_weighted_scores[model_name] = raw_score * m.difficulty_weight

            out.measurements.append(m)

            # Live streaming
            if stream_live and (i + 1) % 10 == 0:
                logger.console.print(f"  [{i + 1}/{len(cases)}] {cid}")

        # Print summary
        if stream_live:
            logger.console.print()
            logger.console.print(f"[highlight]━━━ Weighted Scoring Summary ━━━[/highlight]")
            ws = out.get_weighted_summary()
            dist = ws.get("difficulty_distribution", {})
            logger.console.print(
                f"[muted]Difficulty distribution: easy={dist.get('easy', 0)} "
                f"medium={dist.get('medium', 0)} hard={dist.get('hard', 0)}[/muted]"
            )
            for model_name in model_names:
                model_stats = ws.get("models", {}).get(model_name, {})
                logger.console.print(
                    f"  {model_name}: "
                    f"hard=[metric]{model_stats.get('hard_accuracy', 0) * 100:.1f}%[/metric] │ "
                    f"soft=[metric]{model_stats.get('soft_accuracy', 0) * 100:.1f}%[/metric] │ "
                    f"weighted=[metric]{model_stats.get('weighted_accuracy', 0) * 100:.1f}%[/metric]"
                )
            logger.console.print()

        # Save results
        if output_dir:
            self._save_multi_results(out, output_dir)

        # Optional attention dump for N models
        if bool(getattr(self.config, "dump_attention", False)):
            try:
                dump_attention_multi_model(
                    models=models,
                    cases=cases,
                    case_ids=getattr(self.config, "dump_attention_case_ids", None),
                    benchmark_id=str(benchmark_id),
                    output_dir=output_dir or Path("."),
                    device=self.device,
                    tokenizer_config=self.config.tokenizer,
                    max_tokens=int(getattr(self.config, "dump_attention_max_tokens", 96)),
                    max_heads=int(getattr(self.config, "dump_attention_max_heads", 4)),
                    anchor=str(getattr(self.config, "dump_attention_anchor", "A7")),
                )
            except Exception as e:
                logger.warning(f"Multi-model attention dump failed (continuing): {e!r}")

        return out

    def _save_multi_results(
        self,
        result: BehaviorMultiResult,
        output_dir: Path,
    ) -> None:
        """Save multi-model results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary (comprehensive, includes all data)
        summary_path = output_dir / "behavior_multi_weighted.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.path(str(summary_path), "behavior_multi_weighted")

        # Save comprehensive markdown log for paper/analysis
        md_path = output_dir / "behavior_multi_detailed.md"
        with open(md_path, "w") as f:
            f.write(f"# Behavior Benchmark Results: {result.benchmark_id}\n\n")
            f.write(f"**Models:** {', '.join(result.model_names)}\n\n")
            f.write(f"**Baseline:** {result.baseline_name}\n\n")
            f.write(f"**Total Tests:** {result.total_tests}\n\n")

            # Summary table
            ws = result.get_weighted_summary()
            dist = ws.get("difficulty_distribution", {})
            f.write("## Summary\n\n")
            f.write(
                "**Difficulty Distribution (baseline-defined):** "
                f"Easy={dist.get('easy', 0)}, Medium={dist.get('medium', 0)}, Hard={dist.get('hard', 0)}\n\n"
            )
            f.write(
                "- **Easy (1x)**: baseline EXACT\n"
                "- **Medium (2x)**: baseline CONTAINED\n"
                "- **Hard (3x)**: baseline NONE\n\n"
            )

            # Baseline-failure rescue rates: among baseline-hard cases (baseline NONE),
            # how often does each model get CONTAINED or EXACT?
            hard_cases = [
                m
                for m in result.measurements
                if m.model_match_types.get(result.baseline_name or "", MatchType.NONE) == MatchType.NONE
            ]
            if hard_cases:
                f.write("### Baseline-failure rescue (baseline-hard subset)\n\n")
                f.write(f"- **Baseline-hard cases (baseline NONE)**: {len(hard_cases)}\n\n")
                f.write("| Model | Rescue (soft) | Rescue (exact) |\n")
                f.write("|-------|--------------:|--------------:|\n")
                for model_name in result.model_names:
                    soft_ok = sum(
                        1
                        for m in hard_cases
                        if m.model_match_types.get(model_name) in (MatchType.EXACT, MatchType.CONTAINED)
                    )
                    exact_ok = sum(
                        1 for m in hard_cases if m.model_match_types.get(model_name) == MatchType.EXACT
                    )
                    f.write(
                        f"| {model_name} | {100.0 * soft_ok / len(hard_cases):.1f}% ({soft_ok}/{len(hard_cases)}) | "
                        f"{100.0 * exact_ok / len(hard_cases):.1f}% ({exact_ok}/{len(hard_cases)}) |\n"
                    )
                f.write("\n---\n\n")

            f.write("| Model | Exact | Contained | None | Hard Acc | Soft Acc | Weighted Acc |\n")
            f.write("|-------|-------|-----------|------|----------|----------|-------------|\n")
            models_data = ws.get("models", {})
            for model_name, stats in models_data.items():
                f.write(
                    f"| {model_name} | {stats.get('exact_count', 0)} | "
                    f"{stats.get('contained_count', 0)} | {stats.get('none_count', 0)} | "
                    f"{stats.get('hard_accuracy', 0) * 100:.1f}% | "
                    f"{stats.get('soft_accuracy', 0) * 100:.1f}% | "
                    f"{stats.get('weighted_accuracy', 0) * 100:.1f}% |\n"
                )
            f.write("\n---\n\n")

            # Per-case detailed results
            f.write("## Detailed Results by Case\n\n")
            for i, m in enumerate(result.measurements):
                f.write(f"### Case {i + 1}: `{m.case_id}`\n\n")
                f.write(f"**Difficulty Weight:** {m.difficulty_weight} ({'Easy' if m.difficulty_weight == 1.0 else 'Medium' if m.difficulty_weight == 2.0 else 'Hard'})\n\n")

                f.write("**Prompt:**\n```\n")
                f.write(m.prompt)
                f.write("\n```\n\n")

                f.write(f"**Expected:** `{m.expected}`\n\n")

                f.write("**Model Outputs:**\n\n")
                f.write("| Model | Output | Match | Score | Weighted |\n")
                f.write("|-------|--------|-------|-------|----------|\n")
                for model_name in result.model_names:
                    output = m.model_outputs.get(model_name, "<no output>")
                    match_type = m.model_match_types.get(model_name, MatchType.NONE)
                    raw_score = m.model_raw_scores.get(model_name, 0.0)
                    weighted_score = m.model_weighted_scores.get(model_name, 0.0)
                    # Escape pipes in output for markdown table
                    output_escaped = output.replace("|", "\\|").replace("\n", " ")[:60]
                    if len(output) > 60:
                        output_escaped += "..."
                    f.write(f"| {model_name} | `{output_escaped}` | {match_type.name} | {raw_score} | {weighted_score:.1f} |\n")
                f.write("\n")

                # Full outputs (not truncated)
                f.write("<details>\n<summary>Full Model Outputs</summary>\n\n")
                for model_name in result.model_names:
                    output = m.model_outputs.get(model_name, "<no output>")
                    match_type = m.model_match_types.get(model_name, MatchType.NONE)
                    f.write(f"**{model_name}** ({match_type.name}):\n```\n{output}\n```\n\n")
                f.write("</details>\n\n")
                f.write("---\n\n")

        logger.path(str(md_path), "behavior_multi_markdown")

        # Save detailed human-readable log with full prompts and outputs
        log_path = output_dir / "behavior_multi_detailed.log"
        with open(log_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write(f"BEHAVIOR BENCHMARK (MULTI-MODEL): {result.benchmark_id}\n")
            f.write(f"Models: {', '.join(result.model_names)}\n")
            f.write(f"Baseline: {result.baseline_name}\n")
            f.write("=" * 100 + "\n\n")

            for i, m in enumerate(result.measurements):
                f.write(f"[{i + 1}/{len(result.measurements)}] Case: {m.case_id}\n")
                f.write("-" * 80 + "\n")
                f.write(f"PROMPT:\n{m.prompt}\n")
                f.write("-" * 80 + "\n")
                f.write(f"EXPECTED: {m.expected}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Difficulty weight: {m.difficulty_weight}\n\n")

                for model_name in result.model_names:
                    output = m.model_outputs.get(model_name, "<no output>")
                    match_type = m.model_match_types.get(model_name, MatchType.NONE)
                    raw_score = m.model_raw_scores.get(model_name, 0.0)
                    weighted_score = m.model_weighted_scores.get(model_name, 0.0)

                    status = "✓" if match_type in (MatchType.EXACT, MatchType.CONTAINED) else "✗"
                    f.write(f"  [{status}] {model_name}:\n")
                    f.write(f"      Output: {output!r}\n")
                    f.write(f"      Match: {match_type.name} | Raw: {raw_score} | Weighted: {weighted_score}\n")
                f.write("\n" + "=" * 100 + "\n\n")

            # Summary
            ws = result.get_weighted_summary()
            f.write("SUMMARY\n")
            f.write("=" * 100 + "\n")
            dist = ws.get("difficulty_distribution", {})
            f.write(f"Difficulty distribution: easy={dist.get('easy', 0)} medium={dist.get('medium', 0)} hard={dist.get('hard', 0)}\n\n")
            models_data = ws.get("models", {})
            for model_name, stats in models_data.items():
                f.write(
                    f"  {model_name}: "
                    f"EXACT={stats.get('exact_count', 0)} "
                    f"CONTAINED={stats.get('contained_count', 0)} "
                    f"NONE={stats.get('none_count', 0)} | "
                    f"hard={stats.get('hard_accuracy', 0) * 100:.1f}% "
                    f"soft={stats.get('soft_accuracy', 0) * 100:.1f}% "
                    f"weighted={stats.get('weighted_accuracy', 0) * 100:.1f}%\n"
                )
            f.write("=" * 100 + "\n")
        logger.path(str(log_path), "behavior_multi_detailed_log")

        # Save CSV summary
        csv_path = output_dir / "behavior_multi_weighted.csv"
        ws = result.get_weighted_summary()
        models_data = ws.get("models", {})
        with open(csv_path, "w") as f:
            if models_data:
                # Header
                f.write("model,exact_count,contained_count,none_count,hard_accuracy,soft_accuracy,weighted_accuracy\n")
                for model_name, stats in models_data.items():
                    f.write(
                        f"{model_name},"
                        f"{stats.get('exact_count', 0)},"
                        f"{stats.get('contained_count', 0)},"
                        f"{stats.get('none_count', 0)},"
                        f"{stats.get('hard_accuracy', 0):.4f},"
                        f"{stats.get('soft_accuracy', 0):.4f},"
                        f"{stats.get('weighted_accuracy', 0):.4f}\n"
                    )
        logger.path(str(csv_path), "behavior_multi_csv")

        # Save baseline-failure rescue CSV (high-signal interpretation helper).
        try:
            hard_cases = [
                m
                for m in result.measurements
                if m.model_match_types.get(result.baseline_name or "", MatchType.NONE) == MatchType.NONE
            ]
            rescue_path = output_dir / "behavior_baseline_failure_rescue.csv"
            with open(rescue_path, "w") as f:
                f.write("model,baseline_hard_n,rescue_soft_n,rescue_soft_rate,rescue_exact_n,rescue_exact_rate\n")
                if hard_cases:
                    n = len(hard_cases)
                    for model_name in result.model_names:
                        soft_ok = sum(
                            1
                            for m in hard_cases
                            if m.model_match_types.get(model_name)
                            in (MatchType.EXACT, MatchType.CONTAINED)
                        )
                        exact_ok = sum(
                            1 for m in hard_cases if m.model_match_types.get(model_name) == MatchType.EXACT
                        )
                        f.write(
                            f"{model_name},{n},{soft_ok},{soft_ok / n:.6f},{exact_ok},{exact_ok / n:.6f}\n"
                        )
                else:
                    for model_name in result.model_names:
                        f.write(f"{model_name},0,0,0.0,0,0.0\n")
            logger.path(str(rescue_path), "behavior_baseline_failure_rescue")
        except Exception:
            pass

        # Generate LaTeX table
        latex_path = output_dir / "behavior_multi_table.tex"
        with open(latex_path, "w") as f:
            f.write("% Auto-generated LaTeX table for behavior benchmark results\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Behavioral Test Results (Weighted Scoring)}\n")
            f.write("\\label{tab:behavior-weighted}\n")
            f.write("\\begin{tabular}{lrrrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Model & Exact & Cont. & None & Hard & Soft & Weighted \\\\\n")
            f.write("\\midrule\n")

            # Find best for each metric to bold
            best_hard = max((s.get("hard_accuracy", 0), n) for n, s in models_data.items())[1]
            best_soft = max((s.get("soft_accuracy", 0), n) for n, s in models_data.items())[1]
            best_weighted = max((s.get("weighted_accuracy", 0), n) for n, s in models_data.items())[1]

            for model_name, stats in models_data.items():
                hard_str = f"{stats.get('hard_accuracy', 0) * 100:.1f}\\%"
                soft_str = f"{stats.get('soft_accuracy', 0) * 100:.1f}\\%"
                weighted_str = f"{stats.get('weighted_accuracy', 0) * 100:.1f}\\%"

                if model_name == best_hard:
                    hard_str = f"\\textbf{{{hard_str}}}"
                if model_name == best_soft:
                    soft_str = f"\\textbf{{{soft_str}}}"
                if model_name == best_weighted:
                    weighted_str = f"\\textbf{{{weighted_str}}}"

                f.write(
                    f"{model_name} & {stats.get('exact_count', 0)} & "
                    f"{stats.get('contained_count', 0)} & {stats.get('none_count', 0)} & "
                    f"{hard_str} & {soft_str} & {weighted_str} \\\\\n"
                )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        logger.path(str(latex_path), "behavior_multi_latex")

        # Generate visualizations
        # Convert BehaviorMultiResult to WeightedModelSummary format for visualization
        if models_data:
            try:
                summaries: dict[str, WeightedModelSummary] = {}
                for model_name, stats in models_data.items():
                    # Compute DifficultyBreakdown for charts
                    by_diff: dict[str, DifficultyBreakdown] = {}
                    for diff_name, weight in [("easy", 1.0), ("medium", 2.0), ("hard", 3.0)]:
                        subset = [m for m in result.measurements if m.difficulty_weight == weight]
                        if subset:
                            exact_count = sum(1 for m in subset if m.model_match_types.get(model_name) == MatchType.EXACT)
                            contained_count = sum(1 for m in subset if m.model_match_types.get(model_name) == MatchType.CONTAINED)
                            none_count = len(subset) - exact_count - contained_count
                            weighted_score = sum(m.model_weighted_scores.get(model_name, 0.0) for m in subset)

                            by_diff[diff_name] = DifficultyBreakdown(
                                difficulty=diff_name,
                                weight=weight,
                                count=len(subset),
                                exact_count=exact_count,
                                contained_count=contained_count,
                                none_count=none_count,
                                weighted_score=weighted_score,
                                weighted_max=len(subset) * weight,
                            )

                    # Compute CategoryBreakdown for charts
                    by_cat: dict[str, CategoryBreakdown] = {}
                    all_cats = set(m.category for m in result.measurements if m.category)
                    for cat_name in all_cats:
                        subset = [m for m in result.measurements if m.category == cat_name]
                        if subset:
                            exact_count = sum(1 for m in subset if m.model_match_types.get(model_name) == MatchType.EXACT)
                            contained_count = sum(1 for m in subset if m.model_match_types.get(model_name) == MatchType.CONTAINED)
                            by_cat[cat_name] = CategoryBreakdown(
                                category=cat_name,
                                count=len(subset),
                                exact_count=exact_count,
                                contained_count=contained_count,
                                none_count=len(subset) - exact_count - contained_count,
                            )

                    summaries[model_name] = WeightedModelSummary(
                        model_name=model_name,
                        baseline_name=result.baseline_name or "",
                        total_tests=result.total_tests,
                        exact_count=stats.get("exact_count", 0),
                        contained_count=stats.get("contained_count", 0),
                        none_count=stats.get("none_count", 0),
                        raw_score_sum=float(stats.get("exact_count", 0)) + 0.5 * float(stats.get("contained_count", 0)),
                        raw_score_max=float(result.total_tests),
                        hard_accuracy=stats.get("hard_accuracy", 0.0),
                        soft_accuracy=stats.get("soft_accuracy", 0.0),
                        weighted_score_sum=stats.get("weighted_score_sum", 0.0),
                        weighted_score_max=stats.get("weighted_score_max", 0.0),
                        weighted_accuracy=stats.get("weighted_accuracy", 0.0),
                        by_difficulty=by_diff,
                        by_category=by_cat,
                    )


                viz_paths = generate_all_weighted_visualizations(
                    summaries=summaries,
                    output_dir=output_dir,
                    prefix="behavior_",
                )
                for name, path in viz_paths.items():
                    logger.path(str(path), f"viz_{name}")
            except Exception as e:
                logger.warning(f"Failed to generate weighted visualizations: {e!r}")

    def _dump_attention(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        cases: list[EvalCase],
        measurements: list[BehaviorMeasurement],
        benchmark_id: str,
        output_dir: Path | None,
    ) -> None:
        out_base = Path(output_dir) if output_dir is not None else Path(".")
        out_dir = out_base / "attention_dump" / str(benchmark_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        case_by_id = {str(c.id): c for c in cases}
        meas_by_id = {str(m.case_id): m for m in measurements}

        # Decide which cases to dump.
        requested = getattr(self.config, "dump_attention_case_ids", None)
        if requested:
            case_ids = [str(x) for x in requested]
        else:
            # Default: dump only failures/disagreements (high-signal).
            case_ids = []
            for m in measurements:
                disagree = bool(m.teacher_ok) != bool(m.student_ok)
                any_wrong = (not bool(m.teacher_ok)) or (not bool(m.student_ok))
                if disagree or any_wrong:
                    case_ids.append(str(m.case_id))

        if not case_ids:
            logger.info("Attention dump enabled, but no matching cases selected.")
            return

        tok = TokenizerBuilder().build(self.config.tokenizer)
        max_tokens = int(getattr(self.config, "dump_attention_max_tokens", 96))
        max_heads = int(getattr(self.config, "dump_attention_max_heads", 4))
        anchor = str(getattr(self.config, "dump_attention_anchor", "A7"))

        def _find_subseq(hay: list[int], needle: list[int]) -> int | None:
            if not needle or not hay or len(needle) > len(hay):
                return None
            # Prefer the last match (often the "target" occurrence in copy prompts).
            last: int | None = None
            for i in range(0, len(hay) - len(needle) + 1):
                if hay[i : i + len(needle)] == needle:
                    last = int(i)
            return last

        def _find_answer_span(*, prompt_ids: list[int], expected: object) -> tuple[int, int] | None:
            # Only meaningful if expected exists literally in the prompt token stream.
            exp = str(expected)
            # Try a few whitespace variants (tiktoken/GPT-style tokenizers are space-sensitive).
            cands = []
            cands.append(exp)
            cands.append(exp.strip())
            if exp and not exp.startswith(" "):
                cands.append(" " + exp)
                cands.append(" " + exp.lstrip())
            if exp and exp.startswith(" "):
                cands.append(exp.lstrip())

            hay = list(prompt_ids[: int(max_tokens)])
            for s in cands:
                try:
                    ids = tok.encode(str(s))
                except Exception:
                    continue
                j = _find_subseq(hay, list(ids))
                if j is not None and len(ids) > 0:
                    return int(j), int(j + len(ids) - 1)
            return None

        for cid in case_ids:
            case = case_by_id.get(cid)
            if case is None:
                continue
            meas = meas_by_id.get(cid)
            case_dir = out_dir / str(cid)
            case_dir.mkdir(parents=True, exist_ok=True)

            # Save raw prompt/expected/outcomes for traceability.
            expected = case.answer
            answer_span = None
            meta = {
                "case_id": str(cid),
                "kind": str(getattr(case, "kind", "")),
                "prompt": str(case.prompt),
                "expected": expected,
            }
            if meas is not None:
                meta["teacher_ok"] = bool(meas.teacher_ok)
                meta["student_ok"] = bool(meas.student_ok)
                meta["teacher_answer"] = str(meas.teacher_answer)
                meta["student_answer"] = str(meas.student_answer)

            prompt_ids = tok.encode(case.prompt)
            if not prompt_ids:
                continue

            answer_span = _find_answer_span(prompt_ids=prompt_ids, expected=expected)
            if answer_span is not None:
                a0, a1 = answer_span
                meta["answer_span"] = [int(a0), int(a1)]
            else:
                meta["answer_span"] = None

            (case_dir / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Per-token strings for axis labels / debugging.
            token_strs = [tok.decode([i]) for i in prompt_ids[:max_tokens]]
            # Normalize newlines for readability in JSON consumers.
            token_strs = [s.replace("\n", "\\n") for s in token_strs]
            (case_dir / "tokens.json").write_text(json.dumps(token_strs, indent=2), encoding="utf-8")

            # Pick a split index so we can report "exemplar vs target" attention mass.
            split_idx = _find_anchor_token_index(token_strs, anchor=anchor)
            split = int(split_idx if split_idx is not None else len(token_strs) // 2)

            # Run both models and collect their events.
            events = {}
            summaries = {}
            for tag, model in (("teacher", teacher), ("student", student)):
                model_dir = case_dir / tag
                model_dir.mkdir(parents=True, exist_ok=True)

                _assign_attention_viz_ids(model)
                ctx = TrainingVizContext(
                    enabled=True,
                    step=0,
                    max_tokens=int(max_tokens),
                    max_heads=int(max_heads),
                )
                x = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    m_ctx = cast(ModelWithCtx, model)
                    _ = m_ctx(x, ctx=ctx)

                event = ctx.to_event()
                events[tag] = event
                (model_dir / "attn.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

                # Also write a compact numeric summary per layer:
                # how much the *final* query attends to exemplar region vs target region.
                summary = _attention_mass_summary(
                    event,
                    split=int(split),
                    tokens=token_strs,
                    answer_span=answer_span,
                )
                summaries[tag] = summary
                (model_dir / "mass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

                # Render individual model plots (detailed per-head views).
                _render_attention_pngs(
                    model_dir=model_dir,
                    event=event,
                    tokens=token_strs,
                    split=int(split),
                    answer_span=answer_span,
                    case_id=str(cid),
                    model_tag=str(tag),
                )

                # Write a tiny per-model README + CSV/LaTeX summaries to make interpretation explicit.
                try:
                    _write_attention_case_readme(
                        model_dir=model_dir,
                        case_id=str(cid),
                        model_tag=str(tag),
                        tokens=token_strs,
                        split=int(split),
                        expected=expected,
                        answer_span=answer_span,
                        mass_summary=summary,
                    )
                except Exception:
                    pass

            # Render comparison plots (teacher vs student side-by-side).
            _render_attention_comparison_pngs(
                case_dir=case_dir,
                teacher_event=events["teacher"],
                student_event=events["student"],
                teacher_summary=summaries["teacher"],
                student_summary=summaries["student"],
                tokens=token_strs,
                split=int(split),
                answer_span=answer_span,
                case_id=str(cid),
            )

            # Comparison artifacts (README + LaTeX) for paper/debug transparency.
            try:
                _write_attention_comparison_artifacts(
                    case_dir=case_dir,
                    case_id=str(cid),
                    expected=expected,
                    answer_span=answer_span,
                    tokens=token_strs,
                    teacher_summary=summaries.get("teacher", {}),
                    student_summary=summaries.get("student", {}),
                )
            except Exception:
                pass

            # Optional: copy paper-ready PNGs into a stable directory.
            paper_dir_raw = getattr(self.config, "dump_attention_paper_dir", None)
            paper_tag = getattr(self.config, "dump_attention_paper_tag", None)
            if (
                isinstance(paper_dir_raw, str)
                and paper_dir_raw
                and isinstance(paper_tag, str)
                and paper_tag
            ):
                _copy_attention_paper_pngs(
                    src_dir=case_dir,
                    dst_dir=Path(paper_dir_raw),
                    case_id=str(cid),
                    tag=str(paper_tag),
                )


def dump_attention_multi_model(
    *,
    models: dict[str, nn.Module],
    cases: list[EvalCase],
    case_ids: list[str] | None,
    benchmark_id: str,
    output_dir: Path,
    device: torch.device,
    tokenizer_config: TokenizerConfig,
    max_tokens: int = 96,
    max_heads: int = 4,
    anchor: str = "A7",
) -> dict[str, Path]:
    """Dump attention patterns for N models and generate comparison visualizations.

    This is the N-model extension of _dump_attention() that works with
    arbitrary models instead of just teacher/student pairs.

    Args:
        models: Dict mapping model names to nn.Module instances
        cases: List of EvalCase objects with prompts
        case_ids: Specific case IDs to dump (None = all cases)
        benchmark_id: Benchmark identifier for output directory
        output_dir: Base output directory
        device: Device to run models on
        tokenizer_config: Tokenizer configuration dict
        max_tokens: Maximum tokens to include in attention dump
        max_heads: Maximum heads to visualize per model
        anchor: Anchor string to find split point

    Returns:
        Dict mapping artifact names to file paths.
    """
    out_base = Path(output_dir)
    out_dir = out_base / "attention_dump" / str(benchmark_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_by_id = {str(c.id): c for c in cases}
    tok = TokenizerBuilder().build(tokenizer_config)

    # Determine which cases to process
    if case_ids is None:
        case_ids = list(case_by_id.keys())

    if not case_ids:
        logger.info("No cases selected for attention dump.")
        return {}

    all_artifacts: dict[str, Path] = {}

    for cid in case_ids:
        case = case_by_id.get(cid)
        if case is None:
            continue

        case_dir = out_dir / str(cid)
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save case metadata
        meta = {
            "case_id": str(cid),
            "kind": str(getattr(case, "kind", "")),
            "prompt": str(case.prompt),
            "expected": case.answer,
            "models": list(models.keys()),
        }
        (case_dir / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        prompt_ids = tok.encode(case.prompt)
        if not prompt_ids:
            continue

        # Per-token strings for axis labels
        token_strs = [tok.decode([i]) for i in prompt_ids[:max_tokens]]
        token_strs = [s.replace("\n", "\\n") for s in token_strs]
        (case_dir / "tokens.json").write_text(json.dumps(token_strs, indent=2), encoding="utf-8")

        # Try to find the expected answer token span *in the prompt* (only possible for some tasks).
        def _find_subseq(hay: list[int], needle: list[int]) -> int | None:
            if not needle or not hay or len(needle) > len(hay):
                return None
            last: int | None = None
            for i in range(0, len(hay) - len(needle) + 1):
                if hay[i : i + len(needle)] == needle:
                    last = int(i)
            return last

        expected = case.answer
        answer_span: tuple[int, int] | None = None
        try:
            exp = str(expected)
            cands = [exp, exp.strip()]
            if exp and not exp.startswith(" "):
                cands.append(" " + exp)
                cands.append(" " + exp.lstrip())
            if exp and exp.startswith(" "):
                cands.append(exp.lstrip())
            hay = list(prompt_ids[: int(max_tokens)])
            for s in cands:
                ids = tok.encode(str(s))
                j = _find_subseq(hay, list(ids))
                if j is not None and len(ids) > 0:
                    answer_span = (int(j), int(j + len(ids) - 1))
                    break
        except Exception:
            answer_span = None

        # Find split point (exemplar vs target region)
        split_idx = _find_anchor_token_index(token_strs, anchor=anchor)
        split = int(split_idx if split_idx is not None else len(token_strs) // 2)

        # Run all models and collect attention events
        model_events: dict[str, dict] = {}
        model_summaries: dict[str, dict] = {}

        for model_name, model in models.items():
            model_dir = case_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            _assign_attention_viz_ids(model)
            ctx = TrainingVizContext(
                enabled=True,
                step=0,
                max_tokens=int(max_tokens),
                max_heads=int(max_heads),
            )
            x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            with torch.no_grad():
                m_ctx = cast(ModelWithCtx, model)
                _ = m_ctx(x, ctx=ctx)

            event = ctx.to_event()
            model_events[model_name] = event
            (model_dir / "attn.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

            # Compute attention mass summary
            summary = _attention_mass_summary(
                event,
                split=int(split),
                tokens=token_strs,
                answer_span=answer_span,
            )
            model_summaries[model_name] = summary
            (model_dir / "mass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            # Render individual model plots
            _render_attention_pngs(
                model_dir=model_dir,
                event=event,
                tokens=token_strs,
                split=int(split),
                answer_span=answer_span,
                case_id=str(cid),
                model_tag=str(model_name),
            )

        # Render N-model comparison plots
        artifacts = render_multi_model_attention_comparison(
            model_events=model_events,
            tokens=token_strs,
            split=int(split),
            case_id=str(cid),
            output_dir=case_dir,
            answer_span=answer_span,
            max_heads=max_heads,
        )

        for name, path in artifacts.items():
            all_artifacts[f"{cid}_{name}"] = path

    return all_artifacts


def _assign_attention_viz_ids(model: nn.Module) -> None:
    """Ensure attention layers have stable viz ids/names for ctx recording."""
    try:
        i = 0
        for name, mod in model.named_modules():
            if isinstance(mod, AttentionLayer):
                try:
                    setattr(mod, "_viz_index", int(i))
                    setattr(mod, "_viz_name", str(name))
                except Exception:
                    pass
                i += 1
    except Exception:
        return


def _find_anchor_token_index(tokens: list[str], *, anchor: str) -> int | None:
    """Find the first token index that contains `anchor`."""
    if not anchor:
        return None
    for i, t in enumerate(tokens):
        if anchor in t:
            return int(i)
    # Fallback: try the reconstructed string.
    # Use itertools.accumulate to avoid O(n²) string concatenation.
    from itertools import accumulate
    for i, s in enumerate(accumulate(tokens)):
        if anchor in s:
            return int(i)
    return None


def _attention_mass_summary(
    event: dict[str, Any],
    *,
    split: int,
    tokens: list[str],
    answer_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compute exemplar-vs-target attention mass for the final query token, per layer."""
    out: dict[str, Any] = {
        "split": int(split),
        "anchor_tokens": tokens[max(0, split - 2) : min(len(tokens), split + 3)],
        "answer_span": [int(answer_span[0]), int(answer_span[1])] if answer_span is not None else None,
    }
    layers = event.get("layers", [])
    summaries: list[dict[str, Any]] = []
    if not isinstance(layers, list):
        out["layers"] = summaries
        return out

    for layer in layers:
        try:
            idx = int(layer.get("index", -1))
            name = str(layer.get("name", ""))
            mode = str(layer.get("mode", ""))
            attn = layer.get("attn", None)
            if not isinstance(attn, dict):
                continue
            mats = attn.get("matrices", None)
            if not isinstance(mats, list) or not mats:
                continue
            # mats: list[head][tq][tk]
            # Mean over heads.
            tq = len(mats[0])
            tk = len(mats[0][0]) if tq > 0 else 0
            if tq <= 0 or tk <= 0:
                continue
            # Mean last-row distribution across heads.
            last = [0.0] * tk
            h_count = 0
            for h in mats:
                if not isinstance(h, list) or len(h) != tq:
                    continue
                row = h[-1]
                if not isinstance(row, list) or len(row) != tk:
                    continue
                for j in range(tk):
                    try:
                        last[j] += float(row[j])
                    except Exception:
                        pass
                h_count += 1
            if h_count > 0:
                last = [x / float(h_count) for x in last]

            split2 = max(0, min(int(split), tk))
            mass_exemplar = float(sum(last[:split2]))
            mass_target = float(sum(last[split2:tk]))
            mass_answer = None
            if answer_span is not None:
                a0 = max(0, min(int(answer_span[0]), tk))
                a1 = max(0, min(int(answer_span[1]), tk - 1))
                if a0 <= a1:
                    mass_answer = float(sum(last[a0 : a1 + 1]))
            # Top-k tokens by attention weight (from the mean row).
            topk = 12
            pairs = sorted([(float(w), int(i)) for i, w in enumerate(last)], reverse=True)[:topk]
            tops = [{"i": i, "w": w, "tok": tokens[i] if i < len(tokens) else ""} for (w, i) in pairs]
            top1_i = int(tops[0]["i"]) if tops else -1
            top1_w = float(tops[0]["w"]) if tops else float("nan")
            in_answer = None
            if answer_span is not None and top1_i >= 0:
                in_answer = bool(int(answer_span[0]) <= top1_i <= int(answer_span[1]))

            summaries.append(
                {
                    "index": idx,
                    "name": name,
                    "mode": mode,
                    "tq": int(tq),
                    "tk": int(tk),
                    "mass_exemplar": mass_exemplar,
                    "mass_target": mass_target,
                    "mass_answer": mass_answer,
                    "top1": {"i": top1_i, "w": top1_w, "tok": tokens[top1_i] if 0 <= top1_i < len(tokens) else "", "in_answer": in_answer},
                    "top": tops,
                }
            )
        except Exception:
            continue

    out["layers"] = summaries
    return out


def _render_attention_pngs(
    *,
    model_dir: Path,
    event: dict[str, Any],
    tokens: list[str],
    split: int,
    answer_span: tuple[int, int] | None,
    case_id: str,
    model_tag: str,
) -> None:
    """Render two complementary PNG styles from the dumped attention JSON.

    Style 1 (best for diagnosis): layer × key-position heatmap of the final query token.
    Style 2 (best for presentation): last-layer per-head attention matrices (small grid).
    """
    if plt is None or np is None:
        return
    layers = event.get("layers", [])
    if not isinstance(layers, list) or not layers:
        return

    # Collect per-layer last-row (mean over heads).
    rows: list[Any] = []
    names: list[str] = []
    last_layer_heads: list[Any] | None = None

    for layer in layers:
        attn = layer.get("attn", None)
        if not isinstance(attn, dict):
            continue
        mats = attn.get("matrices", None)
        if not isinstance(mats, list) or not mats:
            continue
        # mats: list[head][tq][tk]
        try:
            head_arrays: list[Any] = []
            for h in mats:
                a = np.asarray(h, dtype=np.float32)
                if a.ndim != 2:
                    continue
                head_arrays.append(a)
            if not head_arrays:
                continue
            m = np.stack(head_arrays, axis=0)  # (H,tq,tk)
            last = m[:, -1, :].mean(axis=0)  # (tk,)
        except Exception:
            continue

        rows.append(last)
        names.append(str(layer.get("name", "")))
        last_layer_heads = head_arrays  # overwritten; ends as last valid layer

    if not rows:
        return

    M = np.stack(rows, axis=0)  # (L,tk)
    L, tk = int(M.shape[0]), int(M.shape[1])
    split2 = max(0, min(int(split), tk))
    gold = "#d4af37"
    a0: int | None = None
    a1: int | None = None
    if answer_span is not None:
        a0 = max(0, min(int(answer_span[0]), tk))
        a1 = max(0, min(int(answer_span[1]), tk - 1))
        if a0 > a1:
            a0 = None
            a1 = None

    # ---- Style 1: layer × key heatmap (final query token) ----
    try:
        fig = plt.figure(figsize=(min(14.0, 0.18 * tk + 4.0), min(10.0, 0.28 * L + 3.0)))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.axvline(split2 - 0.5, color="w", linewidth=1.5, alpha=0.9)
        # Gold vertical bars around the correct answer span (if the answer exists in prompt tokens).
        if a0 is not None and a1 is not None:
            ax.axvline(a0 - 0.5, color=gold, linewidth=3.0, alpha=0.95)
            ax.axvline(a1 + 0.5, color=gold, linewidth=3.0, alpha=0.95)

        # Highlight where attention is going: argmax per layer.
        try:
            peaks = M.argmax(axis=1)  # (L,)
            ys = np.arange(L, dtype=np.float32)
            xs = peaks.astype(np.float32)
            in_ans = None
            if a0 is not None and a1 is not None:
                in_ans = (xs >= float(a0)) & (xs <= float(a1))
            if in_ans is None:
                ax.scatter(xs, ys, s=12, c="white", marker="o", edgecolors="black", linewidths=0.3, alpha=0.9)
            else:
                ax.scatter(xs[~in_ans], ys[~in_ans], s=12, c="#ff4d4d", marker="o", edgecolors="black", linewidths=0.3, alpha=0.9, label="peak (off-answer)")
                ax.scatter(xs[in_ans], ys[in_ans], s=12, c="#2ecc71", marker="o", edgecolors="black", linewidths=0.3, alpha=0.9, label="peak (on-answer)")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        except Exception:
            pass
        # No title: captions/README cover interpretation and this saves vertical space.
        ax.set_xlabel("key position (prompt tokens)")
        ax.set_ylabel("layer (sampled attention modules)")
        # Keep ticks lightweight (big prompts).
        if tk <= 64:
            ax.set_xticks(list(range(tk)))
            ax.set_xticklabels([t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk]], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention weight")
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style1_layer_by_token.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    # ---- Style 1b: last-row line plot (mass vs depth) ----
    try:
        mass_ex = M[:, :split2].sum(axis=1)
        mass_tg = M[:, split2:].sum(axis=1)
        mass_ans = None
        mass_off = None
        if a0 is not None and a1 is not None:
            mass_ans = M[:, int(a0) : int(a1) + 1].sum(axis=1)
            mass_off = 1.0 - mass_ans

        fig = plt.figure(figsize=(10.5, 4.0))
        ax = fig.add_subplot(1, 1, 1)

        # If answer span exists in tokens, prefer "answer vs off-answer" focus view.
        if mass_ans is not None and mass_off is not None:
            mean_ans = float(np.mean(mass_ans)) if len(mass_ans) else 0.0
            ax.plot(
                mass_ans,
                label=f"mass on answer (mean={mean_ans:.2f})",
                linewidth=2.6,
                color="#2ecc71",
            )
            ax.plot(
                mass_off,
                label="mass off answer",
                linewidth=2.0,
                linestyle="--",
                color="#ff4d4d",
                alpha=0.9,
            )
            # Mark peak-focus layer + first half-max layer.
            try:
                peak_i = int(np.argmax(mass_ans))
                peak_v = float(mass_ans[peak_i])
                ax.scatter(
                    [peak_i],
                    [peak_v],
                    s=42,
                    c="#2ecc71",
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=5,
                )
                hm = 0.5 * peak_v
                half_i = int(np.argmax(mass_ans >= hm)) if float(peak_v) > 0 else 0
                ax.axvline(float(half_i), color="#2ecc71", alpha=0.22, linewidth=2.0)
                ax.text(
                    float(half_i) + 0.2,
                    min(0.98, peak_v + 0.03),
                    f"half-max @ L{half_i}",
                    fontsize=8,
                    color="#2ecc71",
                    alpha=0.85,
                )
            except Exception:
                pass
        else:
            ax.plot(mass_ex, label="mass on exemplar region", linewidth=2)
            ax.plot(mass_tg, label="mass on target region", linewidth=2)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("layer (sampled attention modules)")
        ax.set_ylabel("attention mass (final query token)")
        # No title (paper-friendly).
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style1b_mass_by_layer.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    # ---- Style 2: last-layer head grid (tq × tk) ----
    if not last_layer_heads:
        return
    try:
        H = int(len(last_layer_heads))
        # Rank heads by "answer focus" for the final query token if answer span exists.
        order = list(range(H))
        if a0 is not None and a1 is not None:
            try:
                scores = []
                for i, a in enumerate(last_layer_heads):
                    lr = np.asarray(a, dtype=np.float32)
                    if lr.ndim != 2 or lr.shape[0] <= 0 or lr.shape[1] <= 0:
                        scores.append((0.0, i))
                        continue
                    row = lr[-1, :]
                    s = float(np.sum(row[int(a0) : int(a1) + 1]))
                    scores.append((s, i))
                scores.sort(key=lambda t: t[0], reverse=True)
                order = [i for (_, i) in scores]
            except Exception:
                order = list(range(H))

        ncols = min(4, H)
        nrows = int((H + ncols - 1) // ncols)
        fig = plt.figure(figsize=(min(14.0, 3.2 * ncols), min(10.0, 2.8 * nrows)))
        for j, i in enumerate(order):
            a = last_layer_heads[i]
            ax = fig.add_subplot(nrows, ncols, j + 1)
            arr = np.asarray(a, dtype=np.float32)
            im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
            ax.axvline(split2 - 0.5, color="w", linewidth=1.0, alpha=0.9)
            # Gold answer span (if available).
            if a0 is not None and a1 is not None:
                ax.axvline(int(a0) - 0.5, color=gold, linewidth=2.4, alpha=0.95)
                ax.axvline(int(a1) + 0.5, color=gold, linewidth=2.4, alpha=0.95)
            # Mark the peak for the final query token.
            try:
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                    row = arr[-1, :]
                    pk = int(np.argmax(row))
                    in_ans = (
                        (a0 is not None and a1 is not None and int(a0) <= pk <= int(a1))
                        if True
                        else False
                    )
                    c = "#2ecc71" if in_ans else "#ff4d4d"
                    if a0 is None or a1 is None:
                        c = "white"
                    ax.scatter(
                        [pk],
                        [arr.shape[0] - 1],
                        s=26,
                        c=c,
                        marker="o",
                        edgecolors="black",
                        linewidths=0.35,
                        alpha=0.9,
                        zorder=6,
                    )
                    tok = tokens[pk] if 0 <= pk < len(tokens) else ""
                    tok = tok if len(tok) <= 6 else tok[:6] + "…"
                    if a0 is not None and a1 is not None:
                        ans_mass = float(np.sum(row[int(a0) : int(a1) + 1]))
                        ax.set_title(f"h{i} • ans={ans_mass:.2f} • pk={tok}", fontsize=9)
                    else:
                        ax.set_title(f"h{i} • pk={tok}", fontsize=9)
            except Exception:
                ax.set_title(f"head {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        # No suptitle (paper-friendly).
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style2_last_layer_heads.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _render_attention_comparison_pngs(
    *,
    case_dir: Path,
    teacher_event: dict[str, Any],
    student_event: dict[str, Any],
    teacher_summary: dict[str, Any],
    student_summary: dict[str, Any],
    tokens: list[str],
    split: int,
    answer_span: tuple[int, int] | None,
    case_id: str,
) -> None:
    """Render side-by-side comparison plots of teacher vs student attention patterns."""
    if plt is None or np is None:
        return

    # Extract layer-wise attention for both models.
    def _extract_layers(event: dict[str, Any]) -> tuple[list[Any], list[str]]:
        layers = event.get("layers", [])
        if not isinstance(layers, list) or not layers:
            return [], []

        rows: list[Any] = []
        names: list[str] = []

        for layer in layers:
            attn = layer.get("attn", None)
            if not isinstance(attn, dict):
                continue
            mats = attn.get("matrices", None)
            if not isinstance(mats, list) or not mats:
                continue

            try:
                head_arrays: list[Any] = []
                for h in mats:
                    a = np.asarray(h, dtype=np.float32)
                    if a.ndim != 2:
                        continue
                    head_arrays.append(a)
                if not head_arrays:
                    continue
                m = np.stack(head_arrays, axis=0)  # (H,tq,tk)
                last = m[:, -1, :].mean(axis=0)  # (tk,)
            except Exception:
                continue

            rows.append(last)
            names.append(str(layer.get("name", "")))

        return rows, names

    teacher_rows, teacher_names = _extract_layers(teacher_event)
    student_rows, student_names = _extract_layers(student_event)

    if not teacher_rows or not student_rows:
        return

    # Style 1: Side-by-side heatmaps (teacher | student).
    try:
        teacher_M = np.stack(teacher_rows, axis=0)  # (L,tk)
        student_M = np.stack(student_rows, axis=0)  # (L,tk)
        L_t, tk_t = int(teacher_M.shape[0]), int(teacher_M.shape[1])
        L_s, tk_s = int(student_M.shape[0]), int(student_M.shape[1])
        split2 = max(0, min(int(split), min(tk_t, tk_s)))
        gold = "#d4af37"
        a0: int | None = None
        a1: int | None = None
        if answer_span is not None:
            a0 = max(0, min(int(answer_span[0]), min(tk_t, tk_s)))
            a1 = max(0, min(int(answer_span[1]), min(tk_t, tk_s) - 1))
            if a0 > a1:
                a0 = None
                a1 = None

        fig = plt.figure(figsize=(min(28.0, 0.18 * max(tk_t, tk_s) * 2 + 4.0), min(10.0, 0.28 * max(L_t, L_s) + 3.0)))

        # CRITICAL: Use shared normalization for fair visual comparison.
        # Attention weights are probabilities in [0, 1], so normalize to this range.
        vmin, vmax = 0.0, 1.0

        # Teacher heatmap (left).
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(teacher_M, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.axvline(split2 - 0.5, color="w", linewidth=1.5, alpha=0.9)
        if a0 is not None and a1 is not None:
            ax1.axvline(a0 - 0.5, color=gold, linewidth=3.0, alpha=0.95)
            ax1.axvline(a1 + 0.5, color=gold, linewidth=3.0, alpha=0.95)
        # Peak markers (argmax per layer) to show where attention concentrates.
        try:
            peaks = teacher_M.argmax(axis=1)
            ys = np.arange(int(teacher_M.shape[0]), dtype=np.float32)
            xs = peaks.astype(np.float32)
            in_ans = None
            if a0 is not None and a1 is not None:
                in_ans = (xs >= float(a0)) & (xs <= float(a1))
            if in_ans is None:
                ax1.scatter(xs, ys, s=10, c="white", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
            else:
                ax1.scatter(xs[~in_ans], ys[~in_ans], s=10, c="#ff4d4d", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
                ax1.scatter(xs[in_ans], ys[in_ans], s=10, c="#2ecc71", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
        except Exception:
            pass
        # No title (paper-friendly).
        ax1.set_xlabel("key position (prompt tokens)")
        ax1.set_ylabel("layer (sampled attention modules)")
        if tk_t <= 64:
            ax1.set_xticks(list(range(tk_t)))
            ax1.set_xticklabels([t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk_t]], rotation=90, fontsize=7)
        fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02, label="attention weight")

        # Student heatmap (right).
        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(student_M, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        ax2.axvline(split2 - 0.5, color="w", linewidth=1.5, alpha=0.9)
        if a0 is not None and a1 is not None:
            ax2.axvline(a0 - 0.5, color=gold, linewidth=3.0, alpha=0.95)
            ax2.axvline(a1 + 0.5, color=gold, linewidth=3.0, alpha=0.95)
        try:
            peaks = student_M.argmax(axis=1)
            ys = np.arange(int(student_M.shape[0]), dtype=np.float32)
            xs = peaks.astype(np.float32)
            in_ans = None
            if a0 is not None and a1 is not None:
                in_ans = (xs >= float(a0)) & (xs <= float(a1))
            if in_ans is None:
                ax2.scatter(xs, ys, s=10, c="white", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
            else:
                ax2.scatter(xs[~in_ans], ys[~in_ans], s=10, c="#ff4d4d", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
                ax2.scatter(xs[in_ans], ys[in_ans], s=10, c="#2ecc71", marker="o", edgecolors="black", linewidths=0.3, alpha=0.8)
        except Exception:
            pass
        # No title (paper-friendly).
        ax2.set_xlabel("key position (prompt tokens)")
        ax2.set_ylabel("layer (sampled attention modules)")
        if tk_s <= 64:
            ax2.set_xticks(list(range(tk_s)))
            ax2.set_xticklabels([t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk_s]], rotation=90, fontsize=7)
        fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02, label="attention weight")

        # No suptitle (paper-friendly).
        fig.tight_layout()
        fig.savefig(case_dir / "comparison_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _write_attention_case_readme(
    *,
    model_dir: Path,
    case_id: str,
    model_tag: str,
    tokens: list[str],
    split: int,
    expected: object,
    answer_span: tuple[int, int] | None,
    mass_summary: dict[str, Any],
) -> None:
    """Write per-model artifacts explaining scoring/visualizations."""
    model_dir.mkdir(parents=True, exist_ok=True)
    gold_desc = "N/A"
    if answer_span is not None:
        a0, a1 = answer_span
        toks = tokens[int(a0) : int(a1) + 1]
        gold_desc = f"tokens[{a0}:{a1}] = {toks!r}"

    md = []
    md.append(f"# Attention dump • case `{case_id}` • model `{model_tag}`")
    md.append("")
    md.append("## What’s in this folder")
    md.append("- `attn.json`: raw attention matrices (per layer, per head).")
    md.append("- `mass.json`: per-layer summary for the **final query token** (mean over heads).")
    md.append("- `attn_style1_layer_by_token.png`: layer×token heatmap for final query token.")
    md.append(
        "- `attn_style1b_mass_by_layer.png`: final-query attention mass vs depth. If the answer span exists in the prompt, this becomes **mass on answer** vs **mass off answer**, with peak/half-max annotations."
    )
    md.append(
        "- `attn_style2_last_layer_heads.png`: per-head attention matrices for the last sampled layer (heads are sorted by answer focus when possible). Dots mark the final-query peak per head."
    )
    md.append("")
    md.append("## How to read `attn_style1_layer_by_token.png`")
    md.append(f"- **White vertical line**: split index (`split={int(split)}`) between exemplar vs target regions.")
    md.append(f"- **Gold vertical bars**: bracketing the *correct answer token span*, when that span exists in the prompt: {gold_desc}.")
    md.append("- **Dots**: per-layer argmax (peak attention) location for the final query token.")
    md.append("  - Green dot: peak is inside the answer span.")
    md.append("  - Red dot: peak is outside the answer span (attention is being drawn elsewhere).")
    md.append("")
    md.append("## Scoring semantics")
    md.append("- **HARD**: EXACT match (answer span equals expected, no extra prefix/suffix).")
    md.append("- **SOFT**: CONTAINED match (expected appears with limited prefix/suffix tolerance; avoids cases like `1 2 3 4 5 6 7 8` counting as correct for expected `8`).")
    md.append("- **WEIGHTED**: weights cases higher when the baseline struggles (see weighted scoring artifacts in the benchmark output).")
    md.append("")
    md.append("## Notes")
    md.append(f"- expected: `{expected!r}`")
    md.append("")

    (model_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Write a small CSV of per-layer top1 and (optional) mass_on_answer.
    try:
        import csv

        rows = mass_summary.get("layers", [])
        path = model_dir / "attention_layer_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer_index", "name", "mode", "tk", "mass_exemplar", "mass_target", "mass_answer", "top1_i", "top1_w", "top1_tok", "top1_in_answer"])
            for r in rows:
                top1 = r.get("top1", {}) if isinstance(r, dict) else {}
                w.writerow(
                    [
                        r.get("index"),
                        r.get("name"),
                        r.get("mode"),
                        r.get("tk"),
                        r.get("mass_exemplar"),
                        r.get("mass_target"),
                        r.get("mass_answer"),
                        top1.get("i"),
                        top1.get("w"),
                        top1.get("tok"),
                        top1.get("in_answer"),
                    ]
                )
    except Exception:
        pass

    # Small LaTeX table with a few aggregate stats.
    try:
        layers = [r for r in mass_summary.get("layers", []) if isinstance(r, dict)]
        mass_ans = [float(r["mass_answer"]) for r in layers if r.get("mass_answer") is not None]
        top_in = [1.0 for r in layers if isinstance(r.get("top1", None), dict) and bool(r["top1"].get("in_answer", False))]
        n = len(layers)
        mean_mass_ans = sum(mass_ans) / float(len(mass_ans)) if mass_ans else 0.0
        frac_top_in = (sum(top_in) / float(n)) if n > 0 else 0.0
        tex = []
        tex.append(r"% Auto-generated attention summary table")
        tex.append(r"\begin{table}[htbp]")
        tex.append(r"\centering")
        tex.append(r"\small")
        tex.append(rf"\caption{{Attention-to-answer summary for case \texttt{{{case_id}}} ({model_tag}).}}")
        tex.append(r"\begin{tabular}{lr}")
        tex.append(r"\toprule")
        tex.append(r"Metric & Value \\")
        tex.append(r"\midrule")
        tex.append(rf"Mean mass on answer span & {mean_mass_ans:.3f} \\")
        tex.append(rf"Frac. layers with peak on answer & {frac_top_in:.3f} \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")
        tex.append("")
        (model_dir / "attention_summary.tex").write_text("\n".join(tex), encoding="utf-8")
    except Exception:
        pass


def _write_attention_comparison_artifacts(
    *,
    case_dir: Path,
    case_id: str,
    expected: object,
    answer_span: tuple[int, int] | None,
    tokens: list[str],
    teacher_summary: dict[str, Any],
    student_summary: dict[str, Any],
) -> None:
    """Write comparison-level explanation artifacts."""
    case_dir.mkdir(parents=True, exist_ok=True)
    gold_desc = "N/A"
    if answer_span is not None:
        a0, a1 = answer_span
        gold_desc = f"tokens[{a0}:{a1}] = {tokens[int(a0) : int(a1) + 1]!r}"
    md = []
    md.append(f"# Attention comparison • case `{case_id}`")
    md.append("")
    md.append("## Visuals")
    md.append("- `comparison_heatmap.png`: teacher vs student heatmaps (shared color scale).")
    md.append("- `comparison_mass_by_layer.png`: exemplar/target mass curves; when answer span is available, also overlays **answer-mass** curves and shades deltas.")
    md.append("- `comparison_last_layer_heads.png`: last-layer head matrices (teacher row / student row). Columns are sorted by student answer-focus when possible; dots mark final-query peaks.")
    md.append("")
    md.append("## Annotations")
    md.append("- White line: exemplar/target split (anchor-based).")
    md.append(f"- Gold bars: correct answer span (if present): {gold_desc}.")
    md.append("")
    md.append("## Notes")
    md.append(f"- expected: `{expected!r}`")
    md.append("")
    (case_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Also emit a tiny LaTeX snippet for paper appendices.
    tex = []
    tex.append(r"% Auto-generated attention comparison note")
    tex.append(r"\paragraph{Attention comparison.}")
    tex.append(
        rf"Case \texttt{{{case_id}}}: gold bars indicate answer span ({gold_desc.replace('_', r'\_')}). "
        r"Red/green peak markers (per-model plots) indicate whether attention peaks fall on the answer."
    )
    tex.append("")
    (case_dir / "attention_comparison_note.tex").write_text("\n".join(tex), encoding="utf-8")

    # Style 2: Attention mass comparison (line plots on same axes).
    try:
        teacher_M = np.stack(teacher_rows, axis=0)
        student_M = np.stack(student_rows, axis=0)
        split2 = max(0, min(int(split), min(teacher_M.shape[1], student_M.shape[1])))

        teacher_mass_ex = teacher_M[:, :split2].sum(axis=1)
        teacher_mass_tg = teacher_M[:, split2:].sum(axis=1)
        student_mass_ex = student_M[:, :split2].sum(axis=1)
        student_mass_tg = student_M[:, split2:].sum(axis=1)
        a0: int | None = None
        a1: int | None = None
        if answer_span is not None:
            a0 = max(0, min(int(answer_span[0]), min(int(teacher_M.shape[1]), int(student_M.shape[1]))))
            a1 = max(0, min(int(answer_span[1]), min(int(teacher_M.shape[1]), int(student_M.shape[1])) - 1))
            if a0 > a1:
                a0 = None
                a1 = None

        fig = plt.figure(figsize=(10.5, 4.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(teacher_mass_ex, label="Teacher: exemplar region", linewidth=1.6, linestyle="-", color="tab:blue", alpha=0.85)
        ax.plot(teacher_mass_tg, label="Teacher: target region", linewidth=1.6, linestyle="--", color="tab:blue", alpha=0.85)
        ax.plot(student_mass_ex, label="Student: exemplar region", linewidth=1.6, linestyle="-", color="tab:orange", alpha=0.85)
        ax.plot(student_mass_tg, label="Student: target region", linewidth=1.6, linestyle="--", color="tab:orange", alpha=0.85)

        # Add answer-vs-off-answer focus curves if answer span exists in tokens.
        if a0 is not None and a1 is not None:
            t_ans = teacher_M[:, int(a0) : int(a1) + 1].sum(axis=1)
            s_ans = student_M[:, int(a0) : int(a1) + 1].sum(axis=1)
            ax.plot(t_ans, label="Teacher: answer span", linewidth=2.6, color="#1f77b4")
            ax.plot(s_ans, label="Student: answer span", linewidth=2.6, color="#ff7f0e")
            # Shade the delta (green where student focuses more on answer).
            try:
                xs = np.arange(len(t_ans))
                ax.fill_between(xs, t_ans, s_ans, where=(s_ans >= t_ans), color="#2ecc71", alpha=0.12, interpolate=True)
                ax.fill_between(xs, t_ans, s_ans, where=(s_ans < t_ans), color="#ff4d4d", alpha=0.08, interpolate=True)
            except Exception:
                pass
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("layer (sampled attention modules)")
        ax.set_ylabel("attention mass (final query token)")
        # No title (paper-friendly).
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(case_dir / "comparison_mass_by_layer.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    # Style 3: Last layer heads comparison (teacher vs student side-by-side).
    try:
        # Extract last layer heads for both models.
        def _extract_last_layer_heads(event: dict[str, Any]) -> list[Any] | None:
            layers = event.get("layers", [])
            if not isinstance(layers, list) or not layers:
                return None

            for layer in reversed(layers):  # Start from last layer
                attn = layer.get("attn", None)
                if not isinstance(attn, dict):
                    continue
                mats = attn.get("matrices", None)
                if not isinstance(mats, list) or not mats:
                    continue

                head_arrays: list[Any] = []
                for h in mats:
                    a = np.asarray(h, dtype=np.float32)
                    if a.ndim != 2:
                        continue
                    head_arrays.append(a)

                if head_arrays:
                    return head_arrays
            return None

        teacher_heads = _extract_last_layer_heads(teacher_event)
        student_heads = _extract_last_layer_heads(student_event)

        if teacher_heads and student_heads:
            H_t = len(teacher_heads)
            H_s = len(student_heads)
            H_max = max(H_t, H_s)
            # Compute split position from first head's dimensions
            tk_min = min(teacher_heads[0].shape[1] if teacher_heads[0].ndim == 2 else 0,
                         student_heads[0].shape[1] if student_heads[0].ndim == 2 else 0)
            split2 = max(0, min(int(split), tk_min))

            # Optional answer span (for overlays)
            gold = "#d4af37"
            a0 = None
            a1 = None
            if answer_span is not None:
                a0 = max(0, min(int(answer_span[0]), tk_min))
                a1 = max(0, min(int(answer_span[1]), tk_min - 1))
                if a0 > a1:
                    a0 = None
                    a1 = None

            # Sort heads by student "answer mass" (final query token) when possible.
            order = list(range(H_max))
            if a0 is not None and a1 is not None and H_s > 0:
                try:
                    scores = []
                    for i in range(H_s):
                        arr = np.asarray(student_heads[i], dtype=np.float32)
                        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                            scores.append((0.0, i))
                            continue
                        row = arr[-1, :]
                        scores.append((float(np.sum(row[int(a0) : int(a1) + 1])), i))
                    scores.sort(key=lambda t: t[0], reverse=True)
                    order = [i for (_, i) in scores] + [i for i in range(H_max) if i >= H_s]
                    order = order[:H_max]
                except Exception:
                    order = list(range(H_max))

            # Create grid: 2 rows x H_max columns (teacher row, student row)
            fig = plt.figure(figsize=(min(20.0, 3.5 * H_max), 6.0))

            for col, i in enumerate(order):
                # Row 1: Teacher heads
                ax1 = fig.add_subplot(2, H_max, col + 1)
                if i < H_t:
                    arr1 = np.asarray(teacher_heads[i], dtype=np.float32)
                    im1 = ax1.imshow(arr1, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
                    ax1.axvline(split2 - 0.5, color="w", linewidth=0.8, alpha=0.9)
                    if a0 is not None and a1 is not None:
                        ax1.axvline(int(a0) - 0.5, color=gold, linewidth=1.6, alpha=0.95)
                        ax1.axvline(int(a1) + 0.5, color=gold, linewidth=1.6, alpha=0.95)
                    # Mark final-query peak token.
                    try:
                        if arr1.ndim == 2 and arr1.shape[0] > 0 and arr1.shape[1] > 0:
                            row = arr1[-1, :]
                            pk = int(np.argmax(row))
                            in_ans = (a0 is not None and a1 is not None and int(a0) <= pk <= int(a1))
                            c = "#2ecc71" if in_ans else "#ff4d4d"
                            if a0 is None or a1 is None:
                                c = "white"
                            ax1.scatter([pk], [arr1.shape[0] - 1], s=18, c=c, edgecolors="black", linewidths=0.3, alpha=0.85)
                    except Exception:
                        pass
                    if i == 0:
                        ax1.set_ylabel("Teacher", fontsize=9, fontweight="bold")
                    # No title (save vertical space).
                else:
                    ax1.axis("off")
                ax1.set_xticks([])
                ax1.set_yticks([])

                # Row 2: Student heads
                ax2 = fig.add_subplot(2, H_max, H_max + col + 1)
                if i < H_s:
                    arr2 = np.asarray(student_heads[i], dtype=np.float32)
                    im2 = ax2.imshow(arr2, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
                    ax2.axvline(split2 - 0.5, color="w", linewidth=0.8, alpha=0.9)
                    if a0 is not None and a1 is not None:
                        ax2.axvline(int(a0) - 0.5, color=gold, linewidth=1.6, alpha=0.95)
                        ax2.axvline(int(a1) + 0.5, color=gold, linewidth=1.6, alpha=0.95)
                    try:
                        if arr2.ndim == 2 and arr2.shape[0] > 0 and arr2.shape[1] > 0:
                            row = arr2[-1, :]
                            pk = int(np.argmax(row))
                            in_ans = (a0 is not None and a1 is not None and int(a0) <= pk <= int(a1))
                            c = "#2ecc71" if in_ans else "#ff4d4d"
                            if a0 is None or a1 is None:
                                c = "white"
                            ax2.scatter([pk], [arr2.shape[0] - 1], s=18, c=c, edgecolors="black", linewidths=0.3, alpha=0.85)
                    except Exception:
                        pass
                    if i == 0:
                        ax2.set_ylabel("Student", fontsize=9, fontweight="bold")
                else:
                    ax2.axis("off")
                ax2.set_xticks([])
                ax2.set_yticks([])

            # No suptitle (paper-friendly).
            fig.tight_layout()
            fig.savefig(case_dir / "comparison_last_layer_heads.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _copy_attention_paper_pngs(*, src_dir: Path, dst_dir: Path, case_id: str, tag: str) -> None:
    """Copy key PNG artifacts into a stable paper figure directory."""
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        mapping = {
            "comparison_heatmap.png": f"{case_id}_{tag}_comparison_heatmap.png",
            "comparison_mass_by_layer.png": f"{case_id}_{tag}_comparison_mass_by_layer.png",
            "comparison_last_layer_heads.png": f"{case_id}_{tag}_comparison_last_layer_heads.png",
        }
        for src_name, dst_name in mapping.items():
            src = src_dir / src_name
            if not src.exists():
                continue
            dst = dst_dir / dst_name
            shutil.copy2(src, dst)
    except Exception:
        return

