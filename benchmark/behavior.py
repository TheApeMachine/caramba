"""Behavior benchmark: prompt-suite checks for teacher/student.

This is a lightweight "unit test" style benchmark that runs a curated set of
prompt cases with ground-truth evaluation logic and produces:
- per-case outcomes (teacher_ok/student_ok + extracted answers)
- aggregate accuracies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from caramba.config.benchmark import BehaviorBenchmarkConfig
from caramba.config.eval import EvalCase
from caramba.eval.suite import run_eval_verify
from caramba.config.eval import EvalThresholds, EvalVerifyConfig


@dataclass
class BehaviorMeasurement:
    case_id: str
    teacher_ok: bool
    student_ok: bool
    teacher_answer: str
    student_answer: str


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


class BehaviorBenchmark:
    def __init__(self, config: BehaviorBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def _load_cases(self) -> list[EvalCase]:
        path = Path(str(self.config.cases_file))
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("BehaviorBenchmark cases_file must be a non-empty YAML list")
        cases: list[EvalCase] = []
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError("BehaviorBenchmark cases must be dict objects")
            cases.append(EvalCase.model_validate(item))
        return cases

    def run(self, *, teacher: nn.Module, student: nn.Module, benchmark_id: str) -> BehaviorResult:
        # Leverage the existing eval suite implementation.
        cfg = EvalVerifyConfig(
            tokenizer=self.config.tokenizer,
            max_new_tokens=int(self.config.max_new_tokens),
            context_window=self.config.context_window,
            # thresholds are irrelevant for benchmarking (we always record), but required by config
            thresholds=EvalThresholds(min_student_accuracy=0.0, max_accuracy_drop=1.0),
            cases=self._load_cases(),
        )
        summary = run_eval_verify(
            teacher=teacher,
            student=student,
            cfg=cfg,
            device=self.device,
        )
        out = BehaviorResult(benchmark_id=str(benchmark_id))
        for r in summary.results:
            out.measurements.append(
                BehaviorMeasurement(
                    case_id=str(r.case_id),
                    teacher_ok=bool(r.teacher_ok),
                    student_ok=bool(r.student_ok),
                    teacher_answer=str(r.teacher_answer),
                    student_answer=str(r.student_answer),
                )
            )
        return out

