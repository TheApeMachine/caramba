"""Instruction-formatted behavioral benchmark (generation-only).

This benchmark reuses the behavioral_suite_v2 template generator to produce a
randomized (slot-based) suite, but renders each prompt in a strict chat format:

    User: <instruction>

    Assistant:

All evaluation is done via free generation (no logprob-only multiple-choice).
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config.benchmark import BehaviorInstructBenchmarkConfig
from config.eval import EvalCase
from console import logger
from data.tokenizers.builder import TokenizerBuilder
from infer.generate import GenerateConfig, Generator
from research.dba.behavioral_suite_v2 import generate_suite
from research.dba.behavioral_suite_v2.scoring import BehavioralScorer, MatchQuality


def _to_user_assistant_prompt(instruction: str) -> str:
    """Render instruction into the required prompt template."""
    instr = str(instruction).replace("\r\n", "\n").replace("\r", "\n").strip()
    # v2 generator appends "\n\nAnswer:" to prompts; strip trailing anchor for instruction style.
    if instr.endswith("\n\nAnswer:"):
        instr = instr[: -len("\n\nAnswer:")]
    if instr.endswith("Answer:"):
        instr = instr[: -len("Answer:")]
    instr = instr.rstrip()
    # NOTE: Do NOT include a trailing space after "Assistant:".
    # Some base LMs can fall into a greedy whitespace loop when the prompt ends with
    # "Assistant: " (space is tokenized and then repeatedly selected).
    return f"User: {instr}\n\nAssistant:"


def _degeneration_metrics(text: str) -> dict[str, Any]:
    """Cheap string-level degeneration metrics (model-agnostic)."""
    t = str(text).strip()
    if not t:
        return {
            "out_chars": 0,
            "word_count": 0,
            "unique_word_ratio": 0.0,
            "dominant_word": "",
            "dominant_word_frac": 0.0,
            "longest_run": 0,
        }
    words = [w for w in t.replace("\n", " ").split(" ") if w]
    if not words:
        return {
            "out_chars": len(t),
            "word_count": 0,
            "unique_word_ratio": 0.0,
            "dominant_word": "",
            "dominant_word_frac": 0.0,
            "longest_run": 0,
        }
    # Dominant word frequency.
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    dom_word, dom_cnt = max(counts.items(), key=lambda kv: kv[1])
    uniq_ratio = float(len(counts)) / float(len(words)) if words else 0.0
    # Longest consecutive run (by whitespace token).
    longest = 1
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return {
        "out_chars": int(len(t)),
        "word_count": int(len(words)),
        "unique_word_ratio": float(uniq_ratio),
        "dominant_word": str(dom_word),
        "dominant_word_frac": float(dom_cnt / float(len(words))) if words else 0.0,
        "longest_run": int(longest),
    }


@dataclass
class BehaviorInstructResult:
    """Results from instruction-style behavioral evaluation."""

    teacher_summary: dict[str, Any] = field(default_factory=dict)
    student_summary: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)

    teacher_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    student_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    scorer: BehavioralScorer | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_summary": self.teacher_summary,
            "student_summary": self.student_summary,
            "comparison": self.comparison,
            "teacher_by_category": self.teacher_by_category,
            "student_by_category": self.student_by_category,
        }


@dataclass
class BehaviorInstructMultiResult:
    """Results from instruction-style evaluation on N models."""

    model_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_by_category: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    pairwise_comparisons: list[dict[str, Any]] = field(default_factory=list)
    scorer: BehavioralScorer | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_summaries": self.model_summaries,
            "model_by_category": self.model_by_category,
            "pairwise_comparisons": self.pairwise_comparisons,
        }


class BenchmarkBehaviorInstruct:
    """Run instruction-style behavioral evaluation comparing teacher and student."""

    def __init__(self, config: BehaviorInstructBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer = TokenizerBuilder().build(config.tokenizer)

    def run(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        output_dir: Path | None = None,
    ) -> BehaviorInstructResult:
        teacher.eval()
        student.eval()

        logger.info(f"Generating instruction suite (seed={self.config.seed})...")
        suite = generate_suite(
            seed=int(self.config.seed),
            tests_per_category=int(self.config.tests_per_category),
            category_counts=self.config.category_counts,
        )

        tests = suite.tests
        if self.config.categories:
            tests = [t for t in tests if t.category in self.config.categories]
            logger.info(f"Filtered to {len(tests)} tests in categories: {self.config.categories}")
        if self.config.subcategories:
            tests = [t for t in tests if getattr(t, "subcategory", None) in self.config.subcategories]
            logger.info(f"Filtered to {len(tests)} tests in subcategories: {self.config.subcategories}")

        logger.info(f"Running {len(tests)} instruction-formatted behavioral tests...")

        scorer = BehavioralScorer()
        for test in tests:
            prompt = _to_user_assistant_prompt(test.prompt)
            scorer.add_test(test.id, expected=str(test.expected), prompt=prompt)

        transcript_lines: list[str] = []
        transcript_path: Path | None = None
        if output_dir:
            transcript_file = getattr(self.config, "transcript_file", None)
            if isinstance(transcript_file, str) and transcript_file:
                transcript_path = Path(transcript_file)
                if not transcript_path.is_absolute():
                    transcript_path = Path(output_dir) / transcript_path
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_lines.append("Behavior Instruct Raw Transcript")
                transcript_lines.append("=" * 80)
                transcript_lines.append("")

        # Optional degeneration metrics CSV (per-test, per-model).
        degen_rows: list[dict[str, Any]] = []
        degen_path: Path | None = None
        if output_dir:
            p = getattr(self.config, "degeneration_metrics_file", None)
            if isinstance(p, str) and p:
                degen_path = Path(p)
                if not degen_path.is_absolute():
                    degen_path = Path(output_dir) / degen_path
                degen_path.parent.mkdir(parents=True, exist_ok=True)

        for i, test in enumerate(tests):
            if self.config.stream_live and (i + 1) % int(self.config.stream_every) == 0:
                logger.info(f"  [{i+1}/{len(tests)}] {test.id}")

            prompt = _to_user_assistant_prompt(test.prompt)
            prompt_ids = self.tokenizer.encode(prompt)

            teacher_out, teacher_meta = self._generate(model=teacher, prompt_ids=prompt_ids, test=test)
            scorer.add_output(test.id, "teacher", teacher_out)

            student_out, student_meta = self._generate(model=student, prompt_ids=prompt_ids, test=test)
            scorer.add_output(test.id, "student", student_out)

            if degen_path is not None:
                tm = _degeneration_metrics(teacher_out)
                sm = _degeneration_metrics(student_out)
                degen_rows.append(
                    {
                        "test_id": str(test.id),
                        "category": str(getattr(test, "category", "")),
                        "subcategory": str(getattr(test, "subcategory", "")),
                        "model": "teacher",
                        "repetition_penalty": float(teacher_meta.get("repetition_penalty", 1.0)),
                        "max_new_tokens": int(teacher_meta.get("max_new_tokens", int(self.config.max_new_tokens))),
                        **tm,
                    }
                )
                degen_rows.append(
                    {
                        "test_id": str(test.id),
                        "category": str(getattr(test, "category", "")),
                        "subcategory": str(getattr(test, "subcategory", "")),
                        "model": "student",
                        "repetition_penalty": float(student_meta.get("repetition_penalty", 1.0)),
                        "max_new_tokens": int(student_meta.get("max_new_tokens", int(self.config.max_new_tokens))),
                        **sm,
                    }
                )

            if transcript_path is not None:
                transcript_lines.append(f"--- {test.id} ---")
                transcript_lines.append("PROMPT:")
                transcript_lines.append(prompt)
                transcript_lines.append("")
                transcript_lines.append("TEACHER_OUTPUT:")
                transcript_lines.append(teacher_out)
                transcript_lines.append("")
                transcript_lines.append("STUDENT_OUTPUT:")
                transcript_lines.append(student_out)
                transcript_lines.append("")
                transcript_lines.append("")

        teacher_summary = scorer.get_model_summary("teacher")
        student_summary = scorer.get_model_summary("student")
        comparison = scorer.compare_models("teacher", "student")

        teacher_by_cat = self._compute_category_breakdown(scorer, "teacher", tests)
        student_by_cat = self._compute_category_breakdown(scorer, "student", tests)

        result = BehaviorInstructResult(
            teacher_summary=teacher_summary,
            student_summary=student_summary,
            comparison=dict(comparison),
            teacher_by_category=teacher_by_cat,
            student_by_category=student_by_cat,
            scorer=scorer,
        )

        if output_dir:
            self._save_results(result, scorer, Path(output_dir))

            if transcript_path is not None:
                transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
                logger.path(str(transcript_path), "behavior_instruct_transcript")

            if degen_path is not None and degen_rows:
                # Stable column order.
                fieldnames = [
                    "test_id",
                    "category",
                    "subcategory",
                    "model",
                    "repetition_penalty",
                    "max_new_tokens",
                    "out_chars",
                    "word_count",
                    "unique_word_ratio",
                    "dominant_word",
                    "dominant_word_frac",
                    "longest_run",
                ]
                with open(degen_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in degen_rows:
                        w.writerow({k: r.get(k, "") for k in fieldnames})
                logger.path(str(degen_path), "behavior_instruct_degeneration")

            if bool(getattr(self.config, "dump_attention", False)):
                raise RuntimeError(
                    "behavior_instruct: attention dump is not supported after the unified behavior refactor "
                    "(use the unified behavior benchmark instead)."
                )

        return result

    def _generate(self, *, model: nn.Module, prompt_ids: list[int], test: Any) -> tuple[str, dict[str, Any]]:
        # Ensure max_seq_len can hold prompt + generation.
        max_seq_len = max(
            int(self.config.context_window or 2048),
            int(len(prompt_ids) + int(self.config.max_new_tokens) + 8),
        )

        # Base settings from config.
        max_new = int(self.config.max_new_tokens)
        rep = float(getattr(self.config, "repetition_penalty", 1.0))
        temp = 0.0  # keep deterministic unless recommended settings override

        # Optional per-test overrides from v2 template metadata.
        if bool(getattr(self.config, "honor_recommended_settings", False)):
            md = getattr(test, "metadata", None)
            if isinstance(md, dict):
                rs = md.get("recommended_settings")
                if isinstance(rs, dict):
                    if "max_new_tokens" in rs:
                        try:
                            max_new = int(rs["max_new_tokens"])
                        except Exception:
                            pass
                    if "repetition_penalty" in rs:
                        try:
                            rep = float(rs["repetition_penalty"])
                        except Exception:
                            pass
                    if "temperature" in rs:
                        try:
                            temp = float(rs["temperature"])
                        except Exception:
                            pass

        max_seq_len = max(
            int(self.config.context_window or 2048),
            int(len(prompt_ids) + int(max_new) + 8),
        )

        # Tokenize stop strings (token-id suffix sequences).
        stop_sequences: list[list[int]] = []
        stop_token_ids: list[int] = []
        stops = getattr(self.config, "stop", None)
        if isinstance(stops, list):
            for s in stops:
                ss = str(s)
                if not ss:
                    continue
                try:
                    ids = list(self.tokenizer.encode(ss))
                except Exception:
                    ids = []
                if not ids:
                    continue
                stop_sequences.append([int(x) for x in ids])
                if len(ids) == 1:
                    stop_token_ids.append(int(ids[0]))

        gen_cfg = GenerateConfig(
            max_new_tokens=int(max_new),
            temperature=float(temp),
            max_seq_len=int(max_seq_len),
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=float(rep),
            stop_sequences=stop_sequences,
            stop_token_ids=stop_token_ids,
        )

        input_ids = torch.tensor([prompt_ids], device=self.device)
        generator = Generator(model, config=gen_cfg, device=self.device)

        with torch.no_grad():
            out = generator.generate(input_ids)

        # Decode only the completion (exclude prompt tokens).
        completion_ids = out[0, len(prompt_ids) :].tolist()

        # If EOS was generated, strip it from decoded completion text.
        eos = self.tokenizer.eos_token_id
        if eos is not None and completion_ids and int(completion_ids[-1]) == int(eos):
            completion_ids = completion_ids[:-1]

        # Strip trailing stop sequences (if any) for clean scoring/transcripts.
        if stop_sequences and completion_ids:
            for seq in sorted(stop_sequences, key=lambda x: len(x), reverse=True):
                k = len(seq)
                if k > 0 and len(completion_ids) >= k and completion_ids[-k:] == seq:
                    completion_ids = completion_ids[:-k]
                    break

        return str(self.tokenizer.decode(completion_ids)), {
            "max_new_tokens": int(max_new),
            "temperature": float(temp),
            "repetition_penalty": float(rep),
        }

    def run_multi(
        self,
        *,
        models: dict[str, nn.Module],
        output_dir: Path | None = None,
    ) -> BehaviorInstructMultiResult:
        """Run the instruction suite on N models."""
        for m in models.values():
            m.eval()

        logger.info(f"Generating instruction suite (seed={self.config.seed})...")
        suite = generate_suite(
            seed=int(self.config.seed),
            tests_per_category=int(self.config.tests_per_category),
            category_counts=self.config.category_counts,
        )
        tests = suite.tests
        if self.config.categories:
            tests = [t for t in tests if t.category in self.config.categories]
            logger.info(f"Filtered to {len(tests)} tests in categories: {self.config.categories}")

        model_names = list(models.keys())
        logger.info(f"Running {len(tests)} instruction tests on {len(model_names)} models...")

        scorer = BehavioralScorer()
        for test in tests:
            prompt = _to_user_assistant_prompt(test.prompt)
            scorer.add_test(test.id, expected=str(test.expected), prompt=prompt)

        def _prefixed_path(p: Path, prefix: str) -> Path:
            return p.with_name(f"{prefix}{p.name}")

        transcript_lines: list[str] = []
        transcript_path: Path | None = None
        if output_dir:
            transcript_file = getattr(self.config, "transcript_file", None)
            if isinstance(transcript_file, str) and transcript_file:
                base = Path(transcript_file)
                transcript_path = base if base.is_absolute() else (Path(output_dir) / base)
                transcript_path = _prefixed_path(transcript_path, "multi_")
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_lines.append("Behavior Instruct Raw Transcript (Multi-Model)")
                transcript_lines.append("=" * 80)
                transcript_lines.append(f"MODELS: {', '.join(model_names)}")
                transcript_lines.append("")

        for i, test in enumerate(tests):
            if self.config.stream_live and (i + 1) % int(self.config.stream_every) == 0:
                logger.info(f"  [{i+1}/{len(tests)}] {test.id}")
            prompt = _to_user_assistant_prompt(test.prompt)
            prompt_ids = self.tokenizer.encode(prompt)
            outputs_by_model: dict[str, str] = {}
            for model_name, model in models.items():
                out, _meta = self._generate(model=model, prompt_ids=prompt_ids, test=test)
                scorer.add_output(test.id, str(model_name), out)
                outputs_by_model[str(model_name)] = out

            if transcript_path is not None:
                transcript_lines.append(f"--- {test.id} ---")
                transcript_lines.append("PROMPT:")
                transcript_lines.append(prompt)
                transcript_lines.append("")
                for model_name in model_names:
                    transcript_lines.append(f"{model_name}_OUTPUT:")
                    transcript_lines.append(outputs_by_model.get(str(model_name), ""))
                    transcript_lines.append("")
                transcript_lines.append("")

        model_summaries: dict[str, dict[str, Any]] = {}
        model_by_category: dict[str, dict[str, dict[str, Any]]] = {}
        for model_name in model_names:
            model_summaries[model_name] = scorer.get_model_summary(model_name)
            model_by_category[model_name] = self._compute_category_breakdown(
                scorer, model_name, tests
            )

        pairwise: list[dict[str, Any]] = []
        for i, a in enumerate(model_names):
            for b in model_names[i + 1 :]:
                try:
                    pairwise.append(dict(scorer.compare_models(a, b)))
                except Exception:
                    continue

        result = BehaviorInstructMultiResult(
            model_summaries=model_summaries,
            model_by_category=model_by_category,
            pairwise_comparisons=pairwise,
            scorer=scorer,
        )

        if output_dir:
            od = Path(output_dir)
            od.mkdir(parents=True, exist_ok=True)
            summary_path = od / "behavior_instruct_multi_summary.json"
            summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.path(str(summary_path), "behavior_instruct_multi_summary")

            detailed_path = od / "behavior_instruct_multi_detailed.json"
            detailed_path.write_text(json.dumps(scorer.to_dict(), indent=2), encoding="utf-8")
            logger.path(str(detailed_path), "behavior_instruct_multi_detailed")

            if transcript_path is not None:
                transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
                logger.path(str(transcript_path), "behavior_instruct_multi_transcript")

            if bool(getattr(self.config, "dump_attention", False)):
                raise RuntimeError(
                    "behavior_instruct: attention dump is not supported after the unified behavior refactor "
                    "(use the unified behavior benchmark instead)."
                )

        return result

    def run_multi_isolated(
        self,
        *,
        model_names: list[str],
        load_model,
        unload_model,
        output_dir: Path | None = None,
    ) -> BehaviorInstructMultiResult:
        """Run the instruction suite on N models, loading/unloading one at a time."""
        model_names = [str(n) for n in model_names if str(n).strip()]
        if not model_names:
            return BehaviorInstructMultiResult()

        logger.info(f"Generating instruction suite (seed={self.config.seed})...")
        suite = generate_suite(
            seed=int(self.config.seed),
            tests_per_category=int(self.config.tests_per_category),
            category_counts=self.config.category_counts,
        )
        tests = suite.tests
        if self.config.categories:
            tests = [t for t in tests if t.category in self.config.categories]
            logger.info(f"Filtered to {len(tests)} tests in categories: {self.config.categories}")

        scorer = BehavioralScorer()
        for test in tests:
            prompt = _to_user_assistant_prompt(test.prompt)
            scorer.add_test(test.id, expected=str(test.expected), prompt=prompt)

        def _prefixed_path(p: Path, prefix: str) -> Path:
            return p.with_name(f"{prefix}{p.name}")

        transcript_lines: list[str] = []
        transcript_path: Path | None = None
        if output_dir:
            transcript_file = getattr(self.config, "transcript_file", None)
            if isinstance(transcript_file, str) and transcript_file:
                base = Path(transcript_file)
                transcript_path = base if base.is_absolute() else (Path(output_dir) / base)
                transcript_path = _prefixed_path(transcript_path, "isolated_")
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_lines.append("Behavior Instruct Raw Transcript (Isolated Multi-Model)")
                transcript_lines.append("=" * 80)
                transcript_lines.append(f"MODELS: {', '.join(model_names)}")
                transcript_lines.append("")

        for mi, model_name in enumerate(model_names):
            logger.subheader(f"behavior_instruct:model:{model_name} [{mi+1}/{len(model_names)}]")
            model = load_model(model_name)
            model.eval()
            try:
                for i, test in enumerate(tests):
                    if self.config.stream_live and (i + 1) % int(self.config.stream_every) == 0:
                        logger.info(f"  [{i+1}/{len(tests)}] {test.id}")
                    prompt = _to_user_assistant_prompt(test.prompt)
                    prompt_ids = self.tokenizer.encode(prompt)
                    out, _meta = self._generate(model=model, prompt_ids=prompt_ids, test=test)
                    scorer.add_output(test.id, str(model_name), out)
                    if transcript_path is not None:
                        transcript_lines.append(f"--- {test.id} ---")
                        transcript_lines.append("PROMPT:")
                        transcript_lines.append(prompt)
                        transcript_lines.append("")
                        transcript_lines.append(f"{model_name}_OUTPUT:")
                        transcript_lines.append(out)
                        transcript_lines.append("")
                        transcript_lines.append("")
            finally:
                unload_model(model)

        model_summaries: dict[str, dict[str, Any]] = {}
        model_by_category: dict[str, dict[str, dict[str, Any]]] = {}
        for model_name in model_names:
            model_summaries[model_name] = scorer.get_model_summary(model_name)
            model_by_category[model_name] = self._compute_category_breakdown(
                scorer, model_name, tests
            )

        pairwise: list[dict[str, Any]] = []
        for i, a in enumerate(model_names):
            for b in model_names[i + 1 :]:
                try:
                    pairwise.append(dict(scorer.compare_models(a, b)))
                except Exception:
                    continue

        result = BehaviorInstructMultiResult(
            model_summaries=model_summaries,
            model_by_category=model_by_category,
            pairwise_comparisons=pairwise,
            scorer=scorer,
        )

        if output_dir:
            od = Path(output_dir)
            od.mkdir(parents=True, exist_ok=True)
            summary_path = od / "behavior_instruct_multi_summary.json"
            summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.path(str(summary_path), "behavior_instruct_multi_summary")

            detailed_path = od / "behavior_instruct_multi_detailed.json"
            detailed_path.write_text(json.dumps(scorer.to_dict(), indent=2), encoding="utf-8")
            logger.path(str(detailed_path), "behavior_instruct_multi_detailed")

            if transcript_path is not None:
                transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
                logger.path(str(transcript_path), "behavior_instruct_isolated_transcript")

            if bool(getattr(self.config, "dump_attention", False)):
                raise RuntimeError(
                    "behavior_instruct: attention dump is not supported after the unified behavior refactor "
                    "(use the unified behavior benchmark instead)."
                )

        return result

    def _compute_category_breakdown(
        self,
        scorer: BehavioralScorer,
        model_id: str,
        tests: list[Any],
    ) -> dict[str, dict[str, Any]]:
        from collections import defaultdict

        by_category: dict[str, list[Any]] = defaultdict(list)
        for test in tests:
            if test.id in scorer.tests and model_id in scorer.tests[test.id].results:
                by_category[str(test.category)].append(scorer.tests[test.id].results[model_id])

        breakdown: dict[str, dict[str, Any]] = {}
        for category, results in by_category.items():
            n = len(results)
            exact = sum(1 for r in results if r.quality == MatchQuality.EXACT)
            partial = sum(1 for r in results if r.quality == MatchQuality.PARTIAL)
            breakdown[str(category)] = {
                "total": int(n),
                "exact": int(exact),
                "partial": int(partial),
                "none": int(n - exact - partial),
                "exact_rate": float(exact / n) if n > 0 else 0.0,
                "partial_or_better_rate": float((exact + partial) / n) if n > 0 else 0.0,
            }
        return breakdown

    def _save_results(
        self,
        result: BehaviorInstructResult,
        scorer: BehavioralScorer,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "behavior_instruct_summary.json"
        summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.path(str(summary_path), "behavior_instruct_summary")

        detailed_path = output_dir / "behavior_instruct_detailed.json"
        detailed_path.write_text(json.dumps(scorer.to_dict(), indent=2), encoding="utf-8")
        logger.path(str(detailed_path), "behavior_instruct_detailed")

        if self.config.log_file:
            log_path = Path(self.config.log_file)
            if not log_path.is_absolute():
                log_path = output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            lines: list[str] = []
            lines.append("Behavior Instruct Evaluation Log")
            lines.append("=" * 80)
            lines.append("")
            lines.append("Teacher Summary:")
            for k, v in result.teacher_summary.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Student Summary:")
            for k, v in result.student_summary.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Comparison:")
            for k, v in result.comparison.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("=" * 80)
            lines.append("Per-Test Results")
            lines.append("=" * 80)
            lines.append("")

            for test_id, tcase in scorer.tests.items():
                lines.append(f"--- {test_id} ---")
                lines.append(f"Expected: {tcase.expected}")
                lines.append("Prompt:")
                lines.append(str(tcase.prompt))
                lines.append("")
                for model_id, match_result in tcase.results.items():
                    lines.append(f"{model_id}: {match_result.quality.name}")
                    lines.append(match_result.actual)
                    lines.append("")
                lines.append("")

            log_path.write_text("\n".join(lines), encoding="utf-8")
            logger.path(str(log_path), "behavior_instruct_log")

