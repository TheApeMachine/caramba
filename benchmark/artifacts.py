"""Artifact generation for paper-ready outputs.

After benchmarking, we need to present results. This module generates:
- CSV files: Raw data for further analysis
- JSON reports: Structured summary with metadata
- PNG charts: Visual comparisons
- LaTeX tables: Ready for paper inclusion
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .latency import LatencyResult
from .memory import MemoryResult
from .perplexity import PerplexityResult
from .behavior import BehaviorResult
from .accuracy import AccuracyResult
from .context import ContextResult
from .perplexity_microscope import write_microscope
from benchmark.stats import mcnemar_exact_pvalue, paired_bootstrap_delta_ci, wilson_ci
from research.dba.behavioral_suite_v2.weighted_scoring import MatchType


@dataclass
class ExperimentMetadata:
    """Metadata describing the experiment."""

    name: str
    timestamp: str
    manifest_path: str
    teacher_checkpoint: str
    student_config: str
    device: str
    notes: str = ""


@dataclass
class ComparisonSummary:
    """Summary comparing teacher and student model performance."""

    teacher_perplexity: float
    student_perplexity: float
    perplexity_ratio: float

    teacher_tokens_per_sec: float
    student_tokens_per_sec: float
    speedup: float

    teacher_kvcache_bytes_per_token: float
    student_kvcache_bytes_per_token: float
    memory_reduction: float

    @property
    def teacher_kvcache_mb_per_token(self) -> float:
        """Teacher KV-cache size in MB per token (for display)."""
        return self.teacher_kvcache_bytes_per_token / (1024 * 1024)

    @property
    def student_kvcache_mb_per_token(self) -> float:
        """Student KV-cache size in MB per token (for display)."""
        return self.student_kvcache_bytes_per_token / (1024 * 1024)


class ArtifactGenerator:
    """Generates paper-ready artifacts from benchmark results.

    Supports multiple output formats: CSV for data analysis, JSON for
    programmatic access, PNG for figures, and LaTeX for direct paper
    inclusion.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Set up the generator with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        *,
        metadata: ExperimentMetadata,
        teacher_perplexity: PerplexityResult | None = None,
        student_perplexity: PerplexityResult | None = None,
        teacher_latency: LatencyResult | None = None,
        student_latency: LatencyResult | None = None,
        teacher_memory: MemoryResult | None = None,
        student_memory: MemoryResult | None = None,
        teacher_accuracy: AccuracyResult | None = None,
        student_accuracy: AccuracyResult | None = None,
        behavior: BehaviorResult | None = None,
        behavioral_v2: "object | None" = None,  # BehavioralV2Result, avoiding circular import
        teacher_context: ContextResult | None = None,
        student_context: ContextResult | None = None,
        audit: dict | None = None,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate all artifacts and return a dict of paths."""
        formats = formats or ["csv", "json", "png", "latex"]
        generated: dict[str, Path] = {}

        summary: ComparisonSummary | None = None
        if (
            teacher_perplexity is not None
            and student_perplexity is not None
            and teacher_latency is not None
            and student_latency is not None
            and teacher_memory is not None
            and student_memory is not None
            and teacher_memory.kvcache_analysis is not None
            and student_memory.kvcache_analysis is not None
        ):
            summary = self._compute_summary(
                teacher_perplexity=teacher_perplexity,
                student_perplexity=student_perplexity,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
            )

        if "json" in formats:
            path = self._write_json_report(
                metadata,
                summary,
                teacher_perplexity=teacher_perplexity,
                student_perplexity=student_perplexity,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
                teacher_accuracy=teacher_accuracy,
                student_accuracy=student_accuracy,
                behavior=behavior,
                behavioral_v2=behavioral_v2,
                teacher_context=teacher_context,
                student_context=student_context,
                audit=audit,
            )
            generated["report.json"] = path

        if "csv" in formats:
            paths = self._write_csv_files(
                teacher_perplexity=teacher_perplexity,
                student_perplexity=student_perplexity,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
                teacher_accuracy=teacher_accuracy,
                student_accuracy=student_accuracy,
                behavior=behavior,
                teacher_context=teacher_context,
                student_context=student_context,
            )
            generated.update(paths)

        if "png" in formats:
            if summary is None:
                raise RuntimeError("PNG artifacts requested but summary is unavailable (missing ppl/latency/memory results).")
            paths = self._generate_charts(
                summary=summary,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
                teacher_accuracy=teacher_accuracy,
                student_accuracy=student_accuracy,
                behavior=behavior,
                teacher_context=teacher_context,
                student_context=student_context,
            )
            generated.update(paths)

        if "latex" in formats:
            if summary is None:
                raise RuntimeError("LaTeX artifacts requested but summary is unavailable (missing ppl/latency/memory results).")
            path = self._write_latex_tables(metadata, summary)
            generated["tables.tex"] = path
            # Optional detailed behavior table for appendices / paper transparency.
            if behavior is not None and behavior.measurements:
                bpath = self._write_latex_behavior_table(behavior=behavior)
                generated["behavior_cases_table.tex"] = bpath
                spath = self._write_latex_behavior_summary_table(metadata=metadata, behavior=behavior)
                generated[spath.name] = spath

        # Perplexity microscope sidecar (batch-level CSV + summary JSON).
        # This is intentionally independent of the main report to keep `report.json`
        # reasonably sized while still enabling deep inspection.
        try:
            if teacher_perplexity is not None and getattr(teacher_perplexity, "batch_loss_sums", None):
                generated.update(
                    write_microscope(output_dir=self.output_dir, result=teacher_perplexity, prefix="teacher")
                )
            if student_perplexity is not None and getattr(student_perplexity, "batch_loss_sums", None):
                generated.update(
                    write_microscope(output_dir=self.output_dir, result=student_perplexity, prefix="student")
                )
        except Exception:
            # Non-critical: microscope is best-effort.
            pass

        return generated

    # Translation table for LaTeX escaping (more efficient than repeated replace calls).
    _LATEX_ESCAPE_TABLE = str.maketrans({
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    })

    @staticmethod
    def _latex_escape(s: str) -> str:
        """Escape a string for safe inclusion in LaTeX text mode."""
        t = str(s)
        # Normalize whitespace but keep newlines visible.
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\t", " ")
        # Escape backslash first (must be before translate since it uses backslash).
        t = t.replace("\\", r"\textbackslash{}")
        # Use translate for single-char replacements (more efficient).
        t = t.translate(ArtifactGenerator._LATEX_ESCAPE_TABLE)
        # Multi-char replacements still need replace.
        t = t.replace("~", r"\textasciitilde{}")
        t = t.replace("^", r"\textasciicircum{}")
        # Make newlines visible without breaking tables.
        t = t.replace("\n", r"\textbackslash{}n ")
        # Collapse runs of spaces.
        t = " ".join(t.split())
        return t

    def _write_latex_behavior_table(self, *, behavior: BehaviorResult) -> Path:
        """Write a per-case behavior table as LaTeX for paper appendices."""
        path = self.output_dir / "behavior_cases_table.tex"

        def _trunc(s: str, n: int = 64) -> str:
            ss = str(s)
            if len(ss) <= int(n):
                return ss
            return ss[: max(0, int(n) - 1)] + "…"

        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\caption{Behavioral probe results (per case).}")
        lines.append(r"\label{tab:behavior_cases}")
        # Use fixed-width columns for answers to avoid overflow.
        lines.append(r"\begin{tabular}{@{}lccp{0.33\textwidth}p{0.33\textwidth}@{}}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Case} & \textbf{T} & \textbf{S} & \textbf{Teacher output} & \textbf{Student output} \\")
        lines.append(r"\midrule")
        for m in behavior.measurements:
            cid = self._latex_escape(str(m.case_id))
            tok = "1" if bool(m.teacher_ok) else "0"
            sok = "1" if bool(m.student_ok) else "0"
            tout = self._latex_escape(_trunc(str(m.teacher_answer)))
            sout = self._latex_escape(_trunc(str(m.student_answer)))
            # Use \texttt for outputs to preserve punctuation feel.
            lines.append(
                rf"{cid} & {tok} & {sok} & \texttt{{{tout}}} & \texttt{{{sout}}} \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_latex_behavior_summary_table(
        self, *, metadata: ExperimentMetadata, behavior: BehaviorResult
    ) -> Path:
        """Write a HARD/SOFT/WEIGHTED summary table (overall + by category).

        Also writes machine-readable companions (CSV/JSON).
        """
        path = self.output_dir / "behavior_results_100k_generated.tex"

        def _category(case_id: str) -> str:
            cid = str(case_id)
            pref = cid.split("_", 1)[0] if "_" in cid else cid
            m = {
                "copy": "Copy",
                "fewshot": "Few-shot",
                "distractor": "Distractors",
                "reason": "Reasoning",
                "math": "Arithmetic",
                "seq": "Sequences",
                "fact": "World knowledge",
                "semantic": "Semantics",
                "format": "Format",
                "context": "Context",
                "robust": "Robustness",
                "edge": "Edge cases",
                "attention": "Attention probes",
                "instruct": "Instruction",
                "consist": "Consistency",
            }
            return m.get(pref, pref)

        def _is_exact(mt: object | None) -> bool:
            if mt is None:
                return False
            if isinstance(mt, MatchType):
                return mt == MatchType.EXACT
            return str(getattr(mt, "name", mt)) == "EXACT"

        def _is_contained(mt: object | None) -> bool:
            if mt is None:
                return False
            if isinstance(mt, MatchType):
                return mt == MatchType.CONTAINED
            return str(getattr(mt, "name", mt)) == "CONTAINED"

        # Collect per-case rows. `cat` is a string; everything else is numeric.
        rows: list[dict[str, float | str]] = []
        for m in behavior.measurements:
            cat = _category(str(getattr(m, "case_id", "")))
            t_mt = getattr(m, "teacher_match_type", None)
            s_mt = getattr(m, "student_match_type", None)
            dw = float(getattr(m, "difficulty_weight", 1.0) or 1.0)
            t_raw = float(getattr(m, "teacher_raw_score", 0.0) or 0.0)
            s_raw = float(getattr(m, "student_raw_score", 0.0) or 0.0)

            # Fallback raw scores if not set but match types exist.
            if t_raw == 0.0 and _is_exact(t_mt):
                t_raw = 1.0
            if t_raw == 0.0 and _is_contained(t_mt):
                t_raw = 0.5
            if s_raw == 0.0 and _is_exact(s_mt):
                s_raw = 1.0
            if s_raw == 0.0 and _is_contained(s_mt):
                s_raw = 0.5

            rows.append(
                {
                    "cat": str(cat),
                    "t_hard": 1.0 if _is_exact(t_mt) else 0.0,
                    "s_hard": 1.0 if _is_exact(s_mt) else 0.0,
                    "t_soft": 1.0 if (_is_exact(t_mt) or _is_contained(t_mt)) else 0.0,
                    "s_soft": 1.0 if (_is_exact(s_mt) or _is_contained(s_mt)) else 0.0,
                    "dw": dw,
                    "t_w": t_raw * dw,
                    "s_w": s_raw * dw,
                }
            )

        by_cat: dict[str, list[dict[str, float | str]]] = {}
        for r in rows:
            by_cat.setdefault(str(r["cat"]), []).append(r)

        def _fmt_pct(x: float) -> str:
            return f"{100.0 * float(x):.1f}\\%"

        def _acc(xs: list[dict[str, float | str]], key: str) -> float:
            if not xs:
                return 0.0
            return float(sum(float(v[key]) for v in xs)) / float(len(xs))

        def _wacc(xs: list[dict[str, float | str]], key: str) -> float:
            if not xs:
                return 0.0
            num = float(sum(float(v[key]) for v in xs))
            den = float(sum(float(v["dw"]) for v in xs))
            return num / den if den > 0 else 0.0

        cats = sorted(by_cat.keys(), key=lambda c: len(by_cat[c]), reverse=True)
        overall = rows

        baseline_label = "Baseline"
        student_label = "DBA"
        # Use metadata.student_config as a human hint if it looks informative.
        try:
            sc = str(getattr(metadata, "student_config", "")).strip()
            if sc and sc.lower() not in {"student", "dba"}:
                student_label = "DBA"
        except Exception:
            pass

        lines: list[str] = []
        lines.append(r"% Auto-generated. Do not edit by hand.")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\caption{Behavioral benchmark results (auto-generated; HARD/SOFT/WEIGHTED).}")
        lines.append(r"\label{tab:behavior_results_100k}")
        lines.append(r"\begin{tabular}{@{}lrrrrr@{}}")
        lines.append(r"\toprule")
        lines.append(
            rf"\textbf{{Category}} & \textbf{{{baseline_label} hard}} & \textbf{{{student_label} hard}} & "
            rf"\textbf{{{baseline_label} soft}} & \textbf{{{student_label} soft}} & \textbf{{{student_label} weighted}} \\"
        )
        lines.append(r"\midrule")
        lines.append(
            rf"\textbf{{Overall}} & "
            rf"{_fmt_pct(_acc(overall, 't_hard'))} & {_fmt_pct(_acc(overall, 's_hard'))} & "
            rf"{_fmt_pct(_acc(overall, 't_soft'))} & {_fmt_pct(_acc(overall, 's_soft'))} & "
            rf"{_fmt_pct(_wacc(overall, 's_w'))} \\"
        )
        lines.append(r"\midrule")
        for c in cats:
            xs = by_cat[c]
            lines.append(
                rf"{self._latex_escape(c)} & "
                rf"{_fmt_pct(_acc(xs, 't_hard'))} & {_fmt_pct(_acc(xs, 's_hard'))} & "
                rf"{_fmt_pct(_acc(xs, 't_soft'))} & {_fmt_pct(_acc(xs, 's_soft'))} & "
                rf"{_fmt_pct(_wacc(xs, 's_w'))} \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

        # Machine-readable companions (required for auditability).
        (self.output_dir / "behavior_summary.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "teacher_hard": _acc(overall, "t_hard"),
                        "student_hard": _acc(overall, "s_hard"),
                        "teacher_soft": _acc(overall, "t_soft"),
                        "student_soft": _acc(overall, "s_soft"),
                        "student_weighted": _wacc(overall, "s_w"),
                        "n": len(overall),
                    },
                    "by_category": {
                        c: {
                            "teacher_hard": _acc(by_cat[c], "t_hard"),
                            "student_hard": _acc(by_cat[c], "s_hard"),
                            "teacher_soft": _acc(by_cat[c], "t_soft"),
                            "student_soft": _acc(by_cat[c], "s_soft"),
                            "student_weighted": _wacc(by_cat[c], "s_w"),
                            "n": len(by_cat[c]),
                        }
                        for c in cats
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        csv_path = self.output_dir / "behavior_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "category",
                    "n",
                    "baseline_hard",
                    "dba_hard",
                    "baseline_soft",
                    "dba_soft",
                    "dba_weighted",
                ]
            )
            w.writerow(
                [
                    "Overall",
                    len(overall),
                    _acc(overall, "t_hard"),
                    _acc(overall, "s_hard"),
                    _acc(overall, "t_soft"),
                    _acc(overall, "s_soft"),
                    _wacc(overall, "s_w"),
                ]
            )
            for c in cats:
                xs = by_cat[c]
                w.writerow(
                    [
                        c,
                        len(xs),
                        _acc(xs, "t_hard"),
                        _acc(xs, "s_hard"),
                        _acc(xs, "t_soft"),
                        _acc(xs, "s_soft"),
                        _wacc(xs, "s_w"),
                    ]
                )

        return path

    def _compute_summary(
        self,
        teacher_perplexity: PerplexityResult | None,
        student_perplexity: PerplexityResult | None,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
    ) -> ComparisonSummary:
        """Compute comparison summary from individual results."""
        if teacher_perplexity is None or student_perplexity is None:
            raise RuntimeError("ArtifactGenerator: perplexity results are required to compute summary.")
        if teacher_latency is None or student_latency is None:
            raise RuntimeError("ArtifactGenerator: latency results are required to compute summary.")
        if teacher_memory is None or student_memory is None:
            raise RuntimeError("ArtifactGenerator: memory results are required to compute summary.")
        if teacher_memory.kvcache_analysis is None or student_memory.kvcache_analysis is None:
            raise RuntimeError("ArtifactGenerator: kvcache_analysis is required to compute summary.")

        t_ppl = float(teacher_perplexity.perplexity)
        s_ppl = float(student_perplexity.perplexity)
        t_tps = float(teacher_latency.avg_tokens_per_second)
        s_tps = float(student_latency.avg_tokens_per_second)
        t_mem = float(teacher_memory.kvcache_analysis.bytes_per_token_fp16)
        s_mem = float(
            student_memory.kvcache_analysis.bytes_per_token_dba_fp16
            if student_memory.kvcache_analysis.bytes_per_token_dba_fp16 is not None
            else student_memory.kvcache_analysis.bytes_per_token_fp16
        )
        if t_ppl <= 0 or s_ppl <= 0 or t_tps <= 0 or s_tps <= 0 or t_mem <= 0 or s_mem <= 0:
            raise RuntimeError(
                "ArtifactGenerator: summary inputs must be > 0 "
                f"(t_ppl={t_ppl}, s_ppl={s_ppl}, t_tps={t_tps}, s_tps={s_tps}, t_mem={t_mem}, s_mem={s_mem})."
            )

        return ComparisonSummary(
            teacher_perplexity=t_ppl,
            student_perplexity=s_ppl,
            perplexity_ratio=s_ppl / t_ppl if t_ppl > 0 else 0.0,
            teacher_tokens_per_sec=t_tps,
            student_tokens_per_sec=s_tps,
            speedup=s_tps / t_tps if t_tps > 0 else 0.0,
            teacher_kvcache_bytes_per_token=t_mem if t_mem else 0.0,
            student_kvcache_bytes_per_token=s_mem if s_mem else 0.0,
            memory_reduction=t_mem / s_mem if s_mem > 0 else 0.0,
        )

    def _write_json_report(
        self,
        metadata: ExperimentMetadata,
        summary: ComparisonSummary | None,
        *,
        teacher_perplexity: PerplexityResult | None = None,
        student_perplexity: PerplexityResult | None = None,
        teacher_latency: LatencyResult | None = None,
        student_latency: LatencyResult | None = None,
        teacher_memory: MemoryResult | None = None,
        student_memory: MemoryResult | None = None,
        teacher_accuracy: AccuracyResult | None = None,
        student_accuracy: AccuracyResult | None = None,
        behavior: BehaviorResult | None = None,
        behavioral_v2: object | None = None,
        teacher_context: ContextResult | None = None,
        student_context: ContextResult | None = None,
        audit: dict | None = None,
    ) -> Path:
        """Write JSON summary report with metadata and comparison."""
        path = self.output_dir / "report.json"

        def _jsonable(x: object) -> object:
            # Primitives
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            # Paths
            if isinstance(x, Path):
                return str(x)
            # Dataclasses
            # Note: dataclasses.is_dataclass returns true for both instances and classes.
            if is_dataclass(x) and not isinstance(x, type):
                return _jsonable(asdict(cast(Any, x)))
            # Containers
            if isinstance(x, dict):
                return {str(k): _jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_jsonable(v) for v in x]
            # NumPy scalars / arrays if present
            try:
                import numpy as np  # type: ignore

                if isinstance(x, np.generic):
                    return x.item()
                if isinstance(x, np.ndarray):
                    return x.tolist()
            except Exception:
                pass
            raise TypeError(f"Object of type {type(x).__name__} is not JSON-serializable")

        report = {
            "metadata": asdict(metadata),
            "summary": (asdict(summary) if summary is not None else None),
            "results": {
                "perplexity": {
                    "teacher": (asdict(teacher_perplexity) if teacher_perplexity is not None else None),
                    "student": (asdict(student_perplexity) if student_perplexity is not None else None),
                },
                "latency": {
                    "teacher": (asdict(teacher_latency) if teacher_latency is not None else None),
                    "student": (asdict(student_latency) if student_latency is not None else None),
                },
                "memory": {
                    "teacher": (asdict(teacher_memory) if teacher_memory is not None else None),
                    "student": (asdict(student_memory) if student_memory is not None else None),
                },
                "behavioral_v2": (
                    asdict(cast(Any, behavioral_v2))
                    if (behavioral_v2 is not None and is_dataclass(behavioral_v2) and not isinstance(behavioral_v2, type))
                    else behavioral_v2
                ),
            },
            "accuracy": {
                "teacher": (
                    {
                        "micro_accuracy": float(teacher_accuracy.micro_accuracy),
                        "tasks": [asdict(t) for t in teacher_accuracy.tasks],
                    }
                    if teacher_accuracy is not None
                    else None
                ),
                "student": (
                    {
                        "micro_accuracy": float(student_accuracy.micro_accuracy),
                        "tasks": [asdict(t) for t in student_accuracy.tasks],
                    }
                    if student_accuracy is not None
                    else None
                ),
            },
            "behavior": (
                {
                    "benchmark_id": str(behavior.benchmark_id),
                    "teacher_accuracy": float(behavior.teacher_accuracy),
                    "student_accuracy": float(behavior.student_accuracy),
                    "teacher_ci95": None,
                    "student_ci95": None,
                    "student_minus_teacher": None,
                    "cases": [
                        {
                            "case_id": str(m.case_id),
                            "teacher_ok": bool(m.teacher_ok),
                            "student_ok": bool(m.student_ok),
                            "teacher_answer": str(m.teacher_answer),
                            "student_answer": str(m.student_answer),
                        }
                        for m in behavior.measurements
                    ],
                }
                if behavior is not None
                else None
            ),
            "context": {
                "teacher": (
                    {
                        "sweep": [asdict(m) for m in teacher_context.sweep],
                        "decode": [asdict(m) for m in teacher_context.decode],
                    }
                    if teacher_context is not None
                    else None
                ),
                "student": (
                    {
                        "sweep": [asdict(m) for m in student_context.sweep],
                        "decode": [asdict(m) for m in student_context.decode],
                    }
                    if student_context is not None
                    else None
                ),
            },
            "generated_at": datetime.now().isoformat(),
        }

        if audit is not None:
            report["audit"] = audit

        # Add paired CI/significance for behavior if present.
        if behavior is not None and behavior.measurements:
            n = len(behavior.measurements)
            t_ok = sum(1 for m in behavior.measurements if bool(m.teacher_ok))
            s_ok = sum(1 for m in behavior.measurements if bool(m.student_ok))
            t_ci = wilson_ci(int(t_ok), int(n))
            s_ci = wilson_ci(int(s_ok), int(n))
            a = [1.0 if bool(m.student_ok) else 0.0 for m in behavior.measurements]
            b = [1.0 if bool(m.teacher_ok) else 0.0 for m in behavior.measurements]
            dci = paired_bootstrap_delta_ci(a, b)
            b_cnt = sum(1 for m in behavior.measurements if bool(m.teacher_ok) and not bool(m.student_ok))
            c_cnt = sum(1 for m in behavior.measurements if (not bool(m.teacher_ok)) and bool(m.student_ok))
            mp = mcnemar_exact_pvalue(int(b_cnt), int(c_cnt))
            report["behavior"]["teacher_ci95"] = [float(t_ci.low), float(t_ci.high)]
            report["behavior"]["student_ci95"] = [float(s_ci.low), float(s_ci.high)]
            report["behavior"]["student_minus_teacher"] = {
                "delta": float(dci.delta),
                "ci95": [float(dci.low), float(dci.high)],
                "mcnemar_p": float(mp),
                "discordant": {
                    "teacher_yes_student_no": int(b_cnt),
                    "teacher_no_student_yes": int(c_cnt),
                },
            }

        with open(path, "w") as f:
            json.dump(_jsonable(report), f, indent=2)

        return path

    def _write_csv_files(
        self,
        teacher_perplexity: PerplexityResult | None,
        student_perplexity: PerplexityResult | None,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
        teacher_accuracy: AccuracyResult | None,
        student_accuracy: AccuracyResult | None,
        behavior: BehaviorResult | None,
        teacher_context: ContextResult | None,
        student_context: ContextResult | None,
    ) -> dict[str, Path]:
        """Write CSV files with raw benchmark data."""
        paths: dict[str, Path] = {}

        # Perplexity CSV
        if teacher_perplexity or student_perplexity:
            path = self.output_dir / "perplexity.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "perplexity", "loss", "num_tokens"])
                if teacher_perplexity:
                    writer.writerow(
                        [
                            teacher_perplexity.model_name,
                            teacher_perplexity.perplexity,
                            teacher_perplexity.loss,
                            teacher_perplexity.num_tokens,
                        ]
                    )
                if student_perplexity:
                    writer.writerow(
                        [
                            student_perplexity.model_name,
                            student_perplexity.perplexity,
                            student_perplexity.loss,
                            student_perplexity.num_tokens,
                        ]
                    )
            paths["perplexity.csv"] = path

        # Latency CSV
        if teacher_latency or student_latency:
            path = self.output_dir / "latency.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "prompt_len",
                        "gen_len",
                        "batch_size",
                        "prefill_ms",
                        "decode_ms",
                        "total_ms",
                        "tokens_per_sec",
                        "ttft_ms",
                        "use_cache",
                    ]
                )
                for result in [teacher_latency, student_latency]:
                    if result:
                        for m in result.measurements:
                            writer.writerow(
                                [
                                    result.model_name,
                                    m.prompt_len,
                                    m.gen_len,
                                    m.batch_size,
                                    f"{m.prefill_time_ms:.2f}",
                                    f"{m.decode_time_ms:.2f}",
                                    f"{m.total_time_ms:.2f}",
                                    f"{m.tokens_per_second:.2f}",
                                    f"{m.time_to_first_token_ms:.2f}",
                                    getattr(m, "use_cache", False),
                                ]
                            )
            paths["latency.csv"] = path

        # Memory CSV
        if teacher_memory or student_memory:
            path = self.output_dir / "memory.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "seq_len",
                        "batch_size",
                        "quantization",
                        "peak_mb",
                        "kvcache_mb",
                        "model_mb",
                    ]
                )
                for result in [teacher_memory, student_memory]:
                    if result:
                        for m in result.measurements:
                            writer.writerow(
                                [
                                    result.model_name,
                                    m.seq_len,
                                    m.batch_size,
                                    m.quantization,
                                    f"{m.peak_memory_mb:.2f}",
                                    f"{m.kvcache_memory_mb:.2f}",
                                    f"{m.model_memory_mb:.2f}",
                                ]
                            )
            paths["memory.csv"] = path

        # Accuracy CSV (downstream tasks)
        if teacher_accuracy is not None or student_accuracy is not None:
            path = self.output_dir / "accuracy.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "task", "split", "accuracy", "correct", "total"])
                for result in [teacher_accuracy, student_accuracy]:
                    if result is None:
                        continue
                    for t in result.tasks:
                        writer.writerow(
                            [
                                result.model_name,
                                t.task,
                                t.split,
                                f"{float(t.accuracy):.6f}",
                                int(t.correct),
                                int(t.total),
                            ]
                        )
            paths["accuracy.csv"] = path

        # Behavior CSV
        if behavior is not None and behavior.measurements:
            path = self.output_dir / "behavior.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "benchmark_id",
                        "case_id",
                        "teacher_ok",
                        "student_ok",
                        "teacher_answer",
                        "student_answer",
                    ]
                )
                for m in behavior.measurements:
                    writer.writerow(
                        [
                            behavior.benchmark_id,
                            m.case_id,
                            int(bool(m.teacher_ok)),
                            int(bool(m.student_ok)),
                            m.teacher_answer,
                            m.student_answer,
                        ]
                    )
            paths["behavior.csv"] = path

        # Context sweep CSVs
        # Use fields() to get header once, then use getattr() to avoid repeated asdict() calls.
        if teacher_context is not None and teacher_context.sweep:
            path = self.output_dir / "context_sweep_teacher.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                field_names = [fld.name for fld in fields(teacher_context.sweep[0])]
                writer.writerow(field_names)
                for m in teacher_context.sweep:
                    writer.writerow([getattr(m, name) for name in field_names])
            paths["context_sweep_teacher.csv"] = path
        if student_context is not None and student_context.sweep:
            path = self.output_dir / "context_sweep_student.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                field_names = [fld.name for fld in fields(student_context.sweep[0])]
                writer.writerow(field_names)
                for m in student_context.sweep:
                    writer.writerow([getattr(m, name) for name in field_names])
            paths["context_sweep_student.csv"] = path

        if teacher_context is not None and teacher_context.decode:
            path = self.output_dir / "context_decode_teacher.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                field_names = [fld.name for fld in fields(teacher_context.decode[0])]
                writer.writerow(field_names)
                for m in teacher_context.decode:
                    writer.writerow([getattr(m, name) for name in field_names])
            paths["context_decode_teacher.csv"] = path
        if student_context is not None and student_context.decode:
            path = self.output_dir / "context_decode_student.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                field_names = [fld.name for fld in fields(student_context.decode[0])]
                writer.writerow(field_names)
                for m in student_context.decode:
                    writer.writerow([getattr(m, name) for name in field_names])
            paths["context_decode_student.csv"] = path

        # Consolidated context diagnostics (single file): makes it easier to see
        # throughput cliffs vs memory pressure vs context length.
        if (teacher_context is not None and teacher_context.decode) or (
            student_context is not None and student_context.decode
        ):
            path = self.output_dir / "context_diagnostics.csv"
            field_names = [
                "model",
                "context_len",
                "ok",
                # throughput
                "decode_tok_per_s",
                "decode_total_ms",
                "decode_len",
                # prefill
                "prefill_total_s",
                "decode_one_ms",
                # loss/ppl (from sweep)
                "loss",
                "ppl",
                # telemetry
                "rss_mb_before",
                "rss_mb_after",
                "mps_allocated_mb_before",
                "mps_allocated_mb_after",
                "mps_driver_allocated_mb_before",
                "mps_driver_allocated_mb_after",
                "mps_recommended_max_mb",
            ]
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(field_names)

                def _write_rows(ctx: ContextResult) -> None:
                    # Index sweep by ctx_len for loss/ppl/decode_one.
                    sweep_by = {int(m.context_len): m for m in ctx.sweep}
                    for d in sorted(ctx.decode, key=lambda m: int(m.context_len)):
                        s = sweep_by.get(int(d.context_len))
                        writer.writerow(
                            [
                                ctx.model_name,
                                int(d.context_len),
                                int(bool(d.ok)),
                                float(d.decode_tok_per_s),
                                float(d.decode_total_ms),
                                int(d.decode_len),
                                float(d.prefill_total_s),
                                float(getattr(s, "decode_one_ms", float("nan"))) if s is not None else float("nan"),
                                float(getattr(s, "loss", float("nan"))) if s is not None else float("nan"),
                                float(getattr(s, "ppl", float("nan"))) if s is not None else float("nan"),
                                getattr(d, "rss_mb_before", None),
                                getattr(d, "rss_mb_after", None),
                                getattr(d, "mps_allocated_mb_before", None),
                                getattr(d, "mps_allocated_mb_after", None),
                                getattr(d, "mps_driver_allocated_mb_before", None),
                                getattr(d, "mps_driver_allocated_mb_after", None),
                                getattr(d, "mps_recommended_max_mb", None),
                            ]
                        )

                if teacher_context is not None and teacher_context.decode:
                    _write_rows(teacher_context)
                if student_context is not None and student_context.decode:
                    _write_rows(student_context)

            paths["context_diagnostics.csv"] = path

        return paths

    def _generate_charts(
        self,
        summary: ComparisonSummary,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
        teacher_accuracy: AccuracyResult | None,
        student_accuracy: AccuracyResult | None,
        behavior: BehaviorResult | None,
        teacher_context: ContextResult | None,
        student_context: ContextResult | None,
    ) -> dict[str, Path]:
        """Generate PNG charts for visual comparison."""
        paths: dict[str, Path] = {}

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ---------------------------------------------------------------------
        # Paper-compat filenames
        #
        # The LaTeX draft in `artifacts/paper/paper.tex` expects:
        # - perplexity.png
        # - latency_tokens_per_sec.png
        #
        # Keep our richer artifact set (summary.png, latency_vs_context.png, ...)
        # but also emit these compatibility names so runs can be dropped into the
        # paper folder without manual renaming.
        # ---------------------------------------------------------------------

        # Summary bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        models = ["Teacher", "Student (DBA)"]
        values = [summary.teacher_perplexity, summary.student_perplexity]
        colors = ["#3498db", "#e74c3c"]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("Perplexity ↓")
        ax.bar_label(bars, fmt="%.2f")

        ax = axes[1]
        values = [summary.teacher_tokens_per_sec, summary.student_tokens_per_sec]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("Tokens/Second ↑")
        ax.bar_label(bars, fmt="%.0f")

        ax = axes[2]
        values = [
            summary.teacher_kvcache_mb_per_token,
            summary.student_kvcache_mb_per_token,
        ]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("KV-Cache (MB/token) ↓")
        ax.bar_label(bars, fmt="%.6f")

        plt.tight_layout()
        path = self.output_dir / "summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        paths["summary.png"] = path

        # Compatibility: perplexity.png (single-panel).
        if summary.teacher_perplexity > 0.0 or summary.student_perplexity > 0.0:
            fig, ax = plt.subplots(figsize=(6, 4))
            vals = [summary.teacher_perplexity, summary.student_perplexity]
            bars = ax.bar(models, vals, color=colors)
            ax.set_ylabel("Perplexity ↓")
            ax.bar_label(bars, fmt="%.2f")
            ax.grid(True, axis="y", alpha=0.2)
            plt.tight_layout()
            p2 = self.output_dir / "perplexity.png"
            plt.savefig(p2, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            paths["perplexity.png"] = p2

        # Latency vs context length chart
        if teacher_latency and student_latency:
            fig, ax = plt.subplots(figsize=(10, 6))

            all_measurements = (
                teacher_latency.measurements + student_latency.measurements
            )
            if all_measurements:
                batch_sizes = sorted(set(m.batch_size for m in all_measurements))
                gen_lens = sorted(set(m.gen_len for m in all_measurements))

                ref_batch = batch_sizes[0]
                ref_gen_len = gen_lens[len(gen_lens) // 2]

                t_data: dict[int, float] = {}
                s_data: dict[int, float] = {}

                for m in teacher_latency.measurements:
                    if m.batch_size == ref_batch and m.gen_len == ref_gen_len:
                        t_data[m.prompt_len] = m.tokens_per_second

                for m in student_latency.measurements:
                    if m.batch_size == ref_batch and m.gen_len == ref_gen_len:
                        s_data[m.prompt_len] = m.tokens_per_second

                if t_data and s_data:
                    x = sorted(t_data.keys())
                    t_y = [t_data.get(k, 0) for k in x]
                    s_y = [s_data.get(k, 0) for k in x]

                    ax.plot(
                        x, t_y, "o-", label="Teacher", color="#3498db", linewidth=2
                    )
                    ax.plot(
                        x,
                        s_y,
                        "s-",
                        label="Student (DBA)",
                        color="#e74c3c",
                        linewidth=2,
                    )
                    ax.set_xlabel("Prompt Length (tokens)")
                    ax.set_ylabel("Tokens/Second")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    path = self.output_dir / "latency_vs_context.png"
                    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
                    paths["latency_vs_context.png"] = path
                    # Compatibility name used by the paper draft.
                    p2 = self.output_dir / "latency_tokens_per_sec.png"
                    try:
                        p2.write_bytes(path.read_bytes())
                        paths["latency_tokens_per_sec.png"] = p2
                    except Exception:
                        # Non-critical: keep primary chart.
                        pass

            plt.close()

        # Memory scaling chart
        if teacher_memory and student_memory:
            t_analysis = teacher_memory.kvcache_analysis
            s_analysis = student_memory.kvcache_analysis

            if t_analysis and s_analysis:
                fig, ax = plt.subplots(figsize=(10, 6))

                seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
                t_mem = [
                    t_analysis.bytes_per_token_fp16 * s / (1024 * 1024)
                    for s in seq_lens
                ]

                if s_analysis.bytes_per_token_dba_fp16:
                    s_mem = [
                        s_analysis.bytes_per_token_dba_fp16 * s / (1024 * 1024)
                        for s in seq_lens
                    ]
                else:
                    s_mem = [
                        s_analysis.bytes_per_token_fp16 * s / (1024 * 1024)
                        for s in seq_lens
                    ]

                ax.plot(
                    seq_lens, t_mem, "o-", label="Teacher", color="#3498db", linewidth=2
                )
                ax.plot(
                    seq_lens,
                    s_mem,
                    "s-",
                    label="Student (DBA)",
                    color="#e74c3c",
                    linewidth=2,
                )
                ax.set_xlabel("Sequence Length (tokens)")
                ax.set_ylabel("KV-Cache Memory (MB)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale("log", base=2)

                path = self.output_dir / "memory_scaling.png"
                plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
                paths["memory_scaling.png"] = path

                plt.close()

        # Behavior accuracy (single-panel).
        if behavior is not None and behavior.measurements:
            fig, ax = plt.subplots(figsize=(6, 4))
            models = ["Teacher", "Student (DBA)"]
            vals = [float(behavior.teacher_accuracy) * 100.0, float(behavior.student_accuracy) * 100.0]
            colors = ["#3498db", "#e74c3c"]
            bars = ax.bar(models, vals, color=colors)
            ax.set_ylabel("Accuracy (%) ↑")
            ax.set_ylim(0, 100)
            ax.bar_label(bars, fmt="%.1f")
            ax.grid(True, axis="y", alpha=0.2)
            plt.tight_layout()
            path = self.output_dir / "behavior_accuracy.png"
            plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            paths["behavior_accuracy.png"] = path

        # Accuracy by task (grouped bars).
        if (teacher_accuracy is not None and teacher_accuracy.tasks) or (
            student_accuracy is not None and student_accuracy.tasks
        ):
            try:
                # Build aligned task list.
                tmap = {t.task: t for t in (teacher_accuracy.tasks if teacher_accuracy else [])}
                smap = {t.task: t for t in (student_accuracy.tasks if student_accuracy else [])}
                tasks = sorted(set(tmap.keys()) | set(smap.keys()))
                if tasks:
                    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(tasks)), 4))
                    x = list(range(len(tasks)))
                    tw = 0.38
                    tv = [float(tmap[k].accuracy) * 100.0 if k in tmap else 0.0 for k in tasks]
                    sv = [float(smap[k].accuracy) * 100.0 if k in smap else 0.0 for k in tasks]
                    ax.bar([i - tw / 2 for i in x], tv, width=tw, label="Teacher", color="#3498db")
                    ax.bar([i + tw / 2 for i in x], sv, width=tw, label="Student (DBA)", color="#e74c3c")
                    ax.set_xticks(x)
                    ax.set_xticklabels(tasks, rotation=30, ha="right")
                    ax.set_ylabel("Accuracy (%) ↑")
                    ax.set_ylim(0, 100)
                    ax.grid(True, axis="y", alpha=0.2)
                    ax.legend()
                    plt.tight_layout()
                    p = self.output_dir / "accuracy_by_task.png"
                    plt.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02)
                    plt.close()
                    paths["accuracy_by_task.png"] = p
            except Exception:
                # Non-critical.
                pass

        # Context sweep plots (compat with paper names).
        def _plot_context_decode_one(*, ctx_result: ContextResult, name: str) -> Path | None:
            rs = [m for m in ctx_result.sweep if m.ok and float(m.decode_one_ms) == float(m.decode_one_ms)]
            rs = sorted(rs, key=lambda m: int(m.context_len))
            if not rs:
                return None
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([m.context_len for m in rs], [m.decode_one_ms for m in rs], marker="o")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Decode 1 token (ms)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = self.output_dir / f"context_decode_one_ms_{name}.png"
            plt.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            return p

        def _plot_context_decode_tps_compare(
            *, a: ContextResult, b: ContextResult, a_name: str, b_name: str
        ) -> Path | None:
            ra = [m for m in a.decode if m.ok and float(m.decode_tok_per_s) == float(m.decode_tok_per_s)]
            rb = [m for m in b.decode if m.ok and float(m.decode_tok_per_s) == float(m.decode_tok_per_s)]
            if not ra or not rb:
                return None
            ra = sorted(ra, key=lambda m: int(m.context_len))
            rb = sorted(rb, key=lambda m: int(m.context_len))
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.plot([m.context_len for m in ra], [m.decode_tok_per_s for m in ra], marker="o", label=a_name)
            ax.plot([m.context_len for m in rb], [m.decode_tok_per_s for m in rb], marker="o", label=b_name)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Decode throughput (tok/s)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            p = self.output_dir / "compare_context_decode_tok_per_sec.png"
            plt.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            return p

        # Single-model plot for student (used by paper appendix as context_decode_one_ms.png).
        # Prefer student (DBA) if available; else teacher.
        chosen = student_context or teacher_context
        if chosen is not None:
            p = _plot_context_decode_one(ctx_result=chosen, name="student" if chosen is student_context else "teacher")
            if p is not None:
                # Paper-compat fixed name.
                compat = self.output_dir / "context_decode_one_ms.png"
                try:
                    compat.write_bytes(p.read_bytes())
                    paths["context_decode_one_ms.png"] = compat
                except Exception:
                    pass
                paths[p.name] = p

        # Compare plot: decode tok/s vs context.
        if teacher_context is not None and student_context is not None:
            p = _plot_context_decode_tps_compare(
                a=teacher_context,
                b=student_context,
                a_name="Teacher",
                b_name="Student (DBA)",
            )
            if p is not None:
                paths["compare_context_decode_tok_per_sec.png"] = p

        # Diagnostics plot: throughput + memory pressure (MPS) overlay.
        # This helps explain "radical" sweep curves: cliffs often coincide with
        # driver memory approaching recommended max.
        try:
            if teacher_context is not None and student_context is not None:
                ta = [m for m in teacher_context.decode if m.ok]
                tb = [m for m in student_context.decode if m.ok]
                if ta and tb:
                    # Only plot memory series if present (MPS).
                    has_mem = any(getattr(m, "mps_driver_allocated_mb_after", None) is not None for m in (ta + tb))
                    if has_mem:
                        ta = sorted(ta, key=lambda m: int(m.context_len))
                        tb = sorted(tb, key=lambda m: int(m.context_len))
                        fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
                        ax2 = ax1.twinx()

                        x1 = [int(m.context_len) for m in ta]
                        y1 = [float(m.decode_tok_per_s) for m in ta]
                        x2 = [int(m.context_len) for m in tb]
                        y2 = [float(m.decode_tok_per_s) for m in tb]

                        ax1.plot(x1, y1, "o-", color="#3498db", label="Teacher tok/s")
                        ax1.plot(x2, y2, "o-", color="#e74c3c", label="Student (DBA) tok/s")
                        ax1.set_xscale("log", base=2)
                        ax1.set_xlabel("Context length (tokens)")
                        ax1.set_ylabel("Decode throughput (tok/s)")

                        m1 = [getattr(m, "mps_driver_allocated_mb_after", None) for m in ta]
                        m2 = [getattr(m, "mps_driver_allocated_mb_after", None) for m in tb]
                        # Replace None with NaN so matplotlib can skip.
                        m1p = [float(v) if v is not None else float("nan") for v in m1]
                        m2p = [float(v) if v is not None else float("nan") for v in m2]
                        ax2.plot(x1, m1p, "--", color="#3498db", alpha=0.55, label="Teacher MPS driver MB")
                        ax2.plot(x2, m2p, "--", color="#e74c3c", alpha=0.55, label="Student MPS driver MB")
                        ax2.set_ylabel("MPS driver allocated (MB)")

                        # Recommended max (if present anywhere).
                        recs = [
                            float(v)
                            for v in (
                                [getattr(m, "mps_recommended_max_mb", None) for m in ta]
                                + [getattr(m, "mps_recommended_max_mb", None) for m in tb]
                            )
                            if v is not None
                        ]
                        if recs:
                            rmax = max(recs)
                            ax2.axhline(rmax, color="black", alpha=0.2, linewidth=1.0)
                            ax2.text(
                                0.02,
                                0.97,
                                f"recommended max ≈ {rmax:.0f} MB",
                                transform=ax2.transAxes,
                                va="top",
                                fontsize=9,
                                alpha=0.7,
                            )

                        ax1.grid(True, alpha=0.25)
                        # Merge legends.
                        h1, l1 = ax1.get_legend_handles_labels()
                        h2, l2 = ax2.get_legend_handles_labels()
                        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
                        plt.tight_layout()
                        pdiag = self.output_dir / "context_sweep_diagnostics.png"
                        plt.savefig(pdiag, dpi=200, bbox_inches="tight", pad_inches=0.02)
                        plt.close()
                        paths["context_sweep_diagnostics.png"] = pdiag
        except Exception:
            # Non-critical.
            pass

        return paths

    def _write_latex_tables(
        self,
        metadata: ExperimentMetadata,
        summary: ComparisonSummary,
    ) -> Path:
        """Write LaTeX tables for direct paper inclusion."""
        path = self.output_dir / "tables.tex"

        latex = f"""% Auto-generated by Caramba on {datetime.now().isoformat()}
% Experiment: {metadata.name}

\\begin{{table}}[h]
\\centering
\\caption{{DBA Upcycle Results: {metadata.name}}}
\\label{{tab:dba-results}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Teacher}} & \\textbf{{Student (DBA)}} & \\textbf{{Change}} \\\\
\\midrule
Perplexity $\\downarrow$ & {summary.teacher_perplexity:.2f} & {summary.student_perplexity:.2f} & {summary.perplexity_ratio:.2f}$\\times$ \\\\
Throughput (tok/s) $\\uparrow$ & {summary.teacher_tokens_per_sec:.0f} & {summary.student_tokens_per_sec:.0f} & {summary.speedup:.2f}$\\times$ \\\\
KV-Cache (bytes/tok) $\\downarrow$ & {summary.teacher_kvcache_bytes_per_token:.0f} & {summary.student_kvcache_bytes_per_token:.0f} & {summary.memory_reduction:.1f}$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Configuration
% Teacher: {metadata.teacher_checkpoint}
% Student: {metadata.student_config}
% Device: {metadata.device}
"""

        with open(path, "w") as f:
            f.write(latex)

        return path
