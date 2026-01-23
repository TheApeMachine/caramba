from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmark.behavior.attention import dump_attention_multi_model, dump_attention_multi_model_isolated
from benchmark.behavior.artifacts import write_behavior_artifacts
from benchmark.behavior.inference import (
    build_tokenizer,
    tokenizer_vocab_size,
    generate_greedy,
    score_choices_by_logprob,
    score_expected_logprob,
)
from benchmark.behavior.schema import BehaviorSuiteSpec
from benchmark.behavior.scoring import classify_match, match_score
from benchmark.behavior.suite import load_behavior_suite, suite_snapshot
from benchmark.behavior.types import (
    BehaviorResult,
    BehaviorSummary,
    CaseModelOutput,
    CaseResult,
    EvalKind,
    GeneratedCase,
    MatchType,
)


class BehaviorBenchmark:
    """Unified YAML-driven behavior benchmark (single suite, N models)."""

    def __init__(self, *, suite_file: str, tokenizer_config, device: torch.device) -> None:
        self.suite_file = str(suite_file)
        self.tokenizer_config = tokenizer_config
        self.device = device

    def run_multi(
        self,
        *,
        models: dict[str, nn.Module],
        benchmark_id: str,
        output_dir: Path,
        baseline_name: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 32,
        context_window: int | None = None,
        dump_attention: bool = True,
        dump_attention_max_tokens: int | None = 96,
        dump_attention_max_heads: int | None = 4,
        dump_attention_anchor: str | None = None,
        ppl_by_model: dict[str, float] | None = None,
    ) -> BehaviorResult:
        if not models:
            raise ValueError("BehaviorBenchmark.run_multi: no models provided.")
        for m in models.values():
            m.eval()

        # Load and materialize suite.
        spec, cases = load_behavior_suite(self.suite_file, seed_override=seed)
        suite_cfg = suite_snapshot(spec=spec)

        model_names = list(models.keys())
        if baseline_name is None:
            baseline_name = model_names[0]
        baseline_name = str(baseline_name)
        if baseline_name not in models:
            raise RuntimeError(f"Baseline model {baseline_name!r} not found in models.")

        tok = build_tokenizer(self.tokenizer_config)
        vv = tokenizer_vocab_size(tok)

        # Convenience: baseline weights lookup.
        bw = spec.baseline_weights
        dw = spec.difficulty_weights

        results: list[CaseResult] = []

        # Running aggregates per model.
        agg: dict[str, dict[str, float]] = {
            mn: {"n": 0.0, "exact": 0.0, "contained": 0.0, "none": 0.0, "sum": 0.0, "max": 0.0}
            for mn in model_names
        }

        for case in cases:
            prompt_ids = tok.encode(case.prompt)
            if not prompt_ids:
                raise RuntimeError(f"Case {case.id!r} tokenized to empty prompt.")

            # Compute baseline output first (needed for baseline-relative scoring).
            outputs: dict[str, CaseModelOutput] = {}

            # We compute per-model outputs in deterministic order (stable).
            for mn in model_names:
                model = models[mn]
                out_text: str
                choice_details = None
                if case.kind == EvalKind.CHOICE_LOGPROB:
                    if not case.choices or case.correct_index is None:
                        raise RuntimeError(f"{case.id!r}: choice_logprob requires choices + correct_index.")
                    details = score_choices_by_logprob(
                        model=model,
                        device=self.device,
                        prompt_ids=list(prompt_ids),
                        choices=list(case.choices),
                        correct_index=int(case.correct_index),
                        tokenizer=tok,
                        context_window=context_window,
                        valid_vocab_size=vv,
                    )
                    out_text = str(details.picked)
                    choice_details = details
                elif case.kind == EvalKind.GENERATION_GREEDY:
                    out_text = generate_greedy(
                        model=model,
                        device=self.device,
                        prompt_ids=list(prompt_ids),
                        tokenizer=tok,
                        max_new_tokens=int(max_new_tokens),
                        context_window=context_window,
                        valid_vocab_size=vv,
                    )
                else:
                    raise RuntimeError(f"Unsupported eval kind: {case.kind!r}")

                exp_lp = score_expected_logprob(
                    model=model,
                    device=self.device,
                    prompt_ids=list(prompt_ids),
                    expected_text=str(case.expected),
                    tokenizer=tok,
                    context_window=context_window,
                    valid_vocab_size=vv,
                )

                # Match classification + scoring.
                mr = classify_match(
                    output=str(out_text),
                    expected=str(case.expected),
                    prompt=str(case.prompt),
                    allow_contained=bool(case.allow_contained),
                    contained_constraints=list(case.contained_constraints),
                    disallow_contained_if_expected_in_prompt=bool(
                        case.disallow_contained_if_expected_in_prompt
                    ),
                )
                mt = mr.match_type
                raw = float(match_score(mt))
                dweight = float(dw[case.difficulty])

                outputs[mn] = CaseModelOutput(
                    model_name=str(mn),
                    output_text=str(out_text),
                    match_type=mt,
                    raw_score=raw,
                    difficulty_weight=dweight,
                    baseline_weight=1.0,  # filled after we know baseline match type
                    final_score=0.0,  # filled after baseline weight
                    choice_logprob=choice_details,
                    expected_logprob=float(exp_lp),
                )

            baseline_mt = outputs[baseline_name].match_type.value

            # Apply baseline-relative weights and finalize.
            for mn in model_names:
                mo = outputs[mn]
                model_mt = mo.match_type.value
                if model_mt not in bw or baseline_mt not in bw[model_mt]:
                    raise RuntimeError(
                        f"Missing baseline weight for model_mt={model_mt!r} baseline_mt={baseline_mt!r}."
                    )
                base_w = float(bw[model_mt][baseline_mt])
                final = float(mo.raw_score) * float(mo.difficulty_weight) * float(base_w)

                # Max possible is "model EXACT" under the same baseline match type.
                max_w = float(bw["exact"][baseline_mt])
                max_score = 1.0 * float(mo.difficulty_weight) * float(max_w)

                outputs[mn] = CaseModelOutput(
                    model_name=str(mo.model_name),
                    output_text=str(mo.output_text),
                    match_type=mo.match_type,
                    raw_score=float(mo.raw_score),
                    difficulty_weight=float(mo.difficulty_weight),
                    baseline_weight=float(base_w),
                    final_score=float(final),
                    choice_logprob=mo.choice_logprob,
                    expected_logprob=mo.expected_logprob,
                )

                a = agg[mn]
                a["n"] += 1.0
                if mo.match_type == MatchType.EXACT:
                    a["exact"] += 1.0
                elif mo.match_type == MatchType.CONTAINED:
                    a["contained"] += 1.0
                else:
                    a["none"] += 1.0
                a["sum"] += float(final)
                a["max"] += float(max_score)

            results.append(CaseResult(case=case, outputs=outputs))

        # Build summaries
        summaries: dict[str, BehaviorSummary] = {}
        for mn in model_names:
            a = agg[mn]
            n = int(a["n"])
            exact = int(a["exact"])
            contained = int(a["contained"])
            none = int(a["none"])
            score_sum = float(a["sum"])
            score_max = float(a["max"])
            hard = float(exact) / float(n) if n > 0 else 0.0
            soft = float(exact + contained) / float(n) if n > 0 else 0.0
            wacc = float(score_sum) / float(score_max) if score_max > 0 else 0.0

            summaries[mn] = BehaviorSummary(
                model_name=str(mn),
                n=int(n),
                exact=int(exact),
                contained=int(contained),
                none=int(none),
                hard_accuracy=float(hard),
                soft_accuracy=float(soft),
                weighted_accuracy=float(wacc),
                score_sum=float(score_sum),
                score_max=float(score_max),
            )

        out = BehaviorResult(
            suite_id=str(spec.id),
            baseline_name=str(baseline_name),
            cases=list(cases),
            results=list(results),
            summaries=summaries,
            suite_config=suite_cfg,
        )

        # Behavior artifacts (must be complete; raise on any failure).
        behavior_dir = Path(output_dir) / str(benchmark_id) / "behavior"
        behavior_dir.mkdir(parents=True, exist_ok=True)
        write_behavior_artifacts(result=out, output_dir=behavior_dir, ppl_by_model=ppl_by_model)

        # Attention dump is "super important"; default on and strict.
        if dump_attention:
            dump_attention_multi_model(
                models=models,
                cases=list(cases),
                benchmark_id=str(benchmark_id),
                output_dir=Path(output_dir),
                device=self.device,
                tokenizer_config=self.tokenizer_config,
                max_tokens=dump_attention_max_tokens,
                max_heads=dump_attention_max_heads,
                anchor=dump_attention_anchor,
            )

        return out

    def run_multi_isolated(
        self,
        *,
        model_names: list[str],
        load_model,
        unload_model,
        benchmark_id: str,
        output_dir: Path,
        baseline_name: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 32,
        context_window: int | None = None,
        dump_attention: bool = True,
        dump_attention_max_tokens: int | None = 96,
        dump_attention_max_heads: int | None = 4,
        dump_attention_anchor: str | None = None,
        ppl_by_model: dict[str, float] | None = None,
    ) -> BehaviorResult:
        """Run the unified suite loading/unloading one model at a time.

        This supports process isolation / low-memory environments while keeping:
        - paired randomness (single suite materialization)
        - baseline-relative scoring
        - multi-model attention comparison (by accumulating per-model events)
        """
        model_names = [str(n) for n in model_names if str(n).strip()]
        if not model_names:
            raise ValueError("BehaviorBenchmark.run_multi_isolated: no model_names provided.")

        # Load and materialize suite once (paired randomness).
        spec, cases = load_behavior_suite(self.suite_file, seed_override=seed)
        suite_cfg = suite_snapshot(spec=spec)

        if baseline_name is None:
            baseline_name = model_names[0]
        baseline_name = str(baseline_name)
        if baseline_name not in set(model_names):
            raise RuntimeError(f"Baseline model {baseline_name!r} not found in model_names.")

        tok = build_tokenizer(self.tokenizer_config)
        vv = tokenizer_vocab_size(tok)

        bw = spec.baseline_weights
        dw = spec.difficulty_weights

        # First pass: run inference + match classification for each model (no baseline weight yet).
        # Stored as dict[case_id][model_name] -> CaseModelOutput with baseline_weight/final_score placeholders.
        by_case: dict[str, dict[str, CaseModelOutput]] = {str(c.id): {} for c in cases}

        for mn in model_names:
            model = load_model(mn)
            try:
                model.eval()
                for case in cases:
                    prompt_ids = tok.encode(case.prompt)
                    if not prompt_ids:
                        raise RuntimeError(f"Case {case.id!r} tokenized to empty prompt.")

                    out_text: str
                    choice_details = None
                    if case.kind == EvalKind.CHOICE_LOGPROB:
                        if not case.choices or case.correct_index is None:
                            raise RuntimeError(
                                f"{case.id!r}: choice_logprob requires choices + correct_index."
                            )
                        details = score_choices_by_logprob(
                            model=model,
                            device=self.device,
                            prompt_ids=list(prompt_ids),
                            choices=list(case.choices),
                            correct_index=int(case.correct_index),
                            tokenizer=tok,
                            context_window=context_window,
                            valid_vocab_size=vv,
                        )
                        out_text = str(details.picked)
                        choice_details = details
                    elif case.kind == EvalKind.GENERATION_GREEDY:
                        out_text = generate_greedy(
                            model=model,
                            device=self.device,
                            prompt_ids=list(prompt_ids),
                            tokenizer=tok,
                            max_new_tokens=int(max_new_tokens),
                            context_window=context_window,
                            valid_vocab_size=vv,
                        )
                    else:
                        raise RuntimeError(f"Unsupported eval kind: {case.kind!r}")

                    exp_lp = score_expected_logprob(
                        model=model,
                        device=self.device,
                        prompt_ids=list(prompt_ids),
                        expected_text=str(case.expected),
                        tokenizer=tok,
                        context_window=context_window,
                        valid_vocab_size=vv,
                    )

                    mr = classify_match(
                        output=str(out_text),
                        expected=str(case.expected),
                        prompt=str(case.prompt),
                        allow_contained=bool(case.allow_contained),
                        contained_constraints=list(case.contained_constraints),
                        disallow_contained_if_expected_in_prompt=bool(
                            case.disallow_contained_if_expected_in_prompt
                        ),
                    )
                    mt = mr.match_type
                    raw = float(match_score(mt))
                    dweight = float(dw[case.difficulty])

                    by_case[str(case.id)][str(mn)] = CaseModelOutput(
                        model_name=str(mn),
                        output_text=str(out_text),
                        match_type=mt,
                        raw_score=float(raw),
                        difficulty_weight=float(dweight),
                        baseline_weight=1.0,  # filled in pass 2
                        final_score=0.0,  # filled in pass 2
                        choice_logprob=choice_details,
                        expected_logprob=float(exp_lp),
                    )
            finally:
                unload_model(model)

        # Second pass: apply baseline-relative weights, aggregate, and build CaseResult list.
        results: list[CaseResult] = []
        agg: dict[str, dict[str, float]] = {
            mn: {"n": 0.0, "exact": 0.0, "contained": 0.0, "none": 0.0, "sum": 0.0, "max": 0.0}
            for mn in model_names
        }

        for case in cases:
            case_outputs = by_case.get(str(case.id), {})
            if baseline_name not in case_outputs:
                raise RuntimeError(f"{case.id!r}: missing baseline output for {baseline_name!r}.")
            baseline_mt = case_outputs[baseline_name].match_type.value

            finalized: dict[str, CaseModelOutput] = {}
            for mn in model_names:
                if mn not in case_outputs:
                    raise RuntimeError(f"{case.id!r}: missing output for model {mn!r}.")
                mo = case_outputs[mn]
                model_mt = mo.match_type.value
                if model_mt not in bw or baseline_mt not in bw[model_mt]:
                    raise RuntimeError(
                        f"Missing baseline weight for model_mt={model_mt!r} baseline_mt={baseline_mt!r}."
                    )
                base_w = float(bw[model_mt][baseline_mt])
                final = float(mo.raw_score) * float(mo.difficulty_weight) * float(base_w)

                max_w = float(bw["exact"][baseline_mt])
                max_score = 1.0 * float(mo.difficulty_weight) * float(max_w)

                finalized[mn] = CaseModelOutput(
                    model_name=str(mo.model_name),
                    output_text=str(mo.output_text),
                    match_type=mo.match_type,
                    raw_score=float(mo.raw_score),
                    difficulty_weight=float(mo.difficulty_weight),
                    baseline_weight=float(base_w),
                    final_score=float(final),
                    choice_logprob=mo.choice_logprob,
                    expected_logprob=mo.expected_logprob,
                )

                a = agg[mn]
                a["n"] += 1.0
                if mo.match_type == MatchType.EXACT:
                    a["exact"] += 1.0
                elif mo.match_type == MatchType.CONTAINED:
                    a["contained"] += 1.0
                else:
                    a["none"] += 1.0
                a["sum"] += float(final)
                a["max"] += float(max_score)

            results.append(CaseResult(case=case, outputs=finalized))

        summaries: dict[str, BehaviorSummary] = {}
        for mn in model_names:
            a = agg[mn]
            n = int(a["n"])
            exact = int(a["exact"])
            contained = int(a["contained"])
            none = int(a["none"])
            score_sum = float(a["sum"])
            score_max = float(a["max"])
            hard = float(exact) / float(n) if n > 0 else 0.0
            soft = float(exact + contained) / float(n) if n > 0 else 0.0
            wacc = float(score_sum) / float(score_max) if score_max > 0 else 0.0

            summaries[mn] = BehaviorSummary(
                model_name=str(mn),
                n=int(n),
                exact=int(exact),
                contained=int(contained),
                none=int(none),
                hard_accuracy=float(hard),
                soft_accuracy=float(soft),
                weighted_accuracy=float(wacc),
                score_sum=float(score_sum),
                score_max=float(score_max),
            )

        out = BehaviorResult(
            suite_id=str(spec.id),
            baseline_name=str(baseline_name),
            cases=list(cases),
            results=list(results),
            summaries=summaries,
            suite_config=suite_cfg,
        )

        behavior_dir = Path(output_dir) / str(benchmark_id) / "behavior"
        behavior_dir.mkdir(parents=True, exist_ok=True)
        write_behavior_artifacts(result=out, output_dir=behavior_dir, ppl_by_model=ppl_by_model)

        if dump_attention:
            dump_attention_multi_model_isolated(
                model_names=list(model_names),
                load_model=load_model,
                unload_model=unload_model,
                cases=list(cases),
                benchmark_id=str(benchmark_id),
                output_dir=Path(output_dir),
                device=self.device,
                tokenizer_config=self.tokenizer_config,
                max_tokens=dump_attention_max_tokens,
                max_heads=dump_attention_max_heads,
                anchor=dump_attention_anchor,
            )

        return out

